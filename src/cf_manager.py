import torch
import os
import sys
import re
import logging
from os.path import isfile
import copy
import threading
import time
import enum
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
try:
		set_start_method('spawn')
except RuntimeError:
		pass


class CFMode(enum.Enum):
	MANUAL = 0
	AUTO = 1


"""
Auto aggressive checkpoint manager for PyTorch

Usage : In the training script:

	Initialize with the model and optimizer in local_rank 0
	chk = CFCheckpoint(model=model, optim=optimizer, dl=dl)
	chk_manager = CFManager(chk, chk_dir='./chk/', freq=CheckFreqMode.AUTO)

	To initiate a checkpoint at the given frequency (done internally if AUTO, 
						if MANUAL, user code must trigger):
	Snapshot the state in-memory
	--------------------------
	chk_manager.snapshot()

	Persist the in-memory snapshot in background
	--------------------------
	chk_manager.persist()

	On recovery, to resume from a checkpoint:
	Restores the latest checkpoint in the dir being managed 
	----------------------------------------
	chk = CFCheckpoint(model=model, optim=optimizer, dl=dl)
	chk_manager = CFManager(chk, chk_dir='./chk/', freq=CheckFreqMode.AUTO)
	chk_manager.restore()

"""


class CFManager:

		"""
		`chk_dir` : Directory where the checkpoints are managed. Can be
				local storage path, or any remote storage that exposes POSIX.
				All checkpoint versions are maintained in this directory, and
				on start-up, the latest checkpoint version in this dir
				is restored.

		`chk` : An instance of the CFCheckpoint class that tracks 
				one or more tractable object for snapshotting.

		`overwrite` : If true, maintains only the latest 
				version of the checkpoint at any instant, with the
				exception of checkpoints made at epoch boundaries
				(controlled by `keep_epoch_chk`). Storage space required
				is low.
				If false, keeps all versions of checkpoints written so far, 
				managed by the checkpointing frequency. Uses a lot of
				storage space -TODO: control this by tuning `keep_latest_n`

				At any point, there can be a max of two active checkpoints
				if `overwrite` is True - one completed, and one ongoing.

		`keep_epoch_chk` : If true, keeps a version of the checkpoint
				taken at epoch boundaries that can be used later to restore
				the model to the best val accuracy for instance. This 
				is the default behaviour. Although beware this assumes you have
				enough storage space to maintain `n` versions of the checkpoint,
				where `n` is the number of epochs you train for.
				If false, checkpoints are overwritten - at any instant, only the
				latest checkpoint is maintained.
				Default is to overwrite iter chk and keep epoch chk.

		`chk_prefix` : Prefix for the cjheckpoint file

		"""

		def __init__(
				self,
				chk_dir,
				chk,
				keep_epoch_chk = True,
				overwrite = True,
				mode = CFMode.AUTO,
				chk_prefix = 'model_v_'):

				self.logger = logging.getLogger(__name__)
				self.chk_dir = chk_dir
				self.chk = chk
				self.keep_epoch_chk = keep_epoch_chk
				self.overwrite = overwrite
				self.chk_prefix = chk_prefix
				self.mode = mode
				self.chk_epoch_subdir = 'epoch'
				self.mp_manager = Manager()
				self.snapshot_copy = None

				self.cpu_side = False	
				# Active snapshot, if true, don't snapshot again
				self.active_snapshot = Value('i', 0)
				self.lock = Lock()
				self.in_progress_snapshot = Value('i', 0)

				# Handle to the process performing checkpoint
				# Can be only one at any instant. A new checkpoint
				# cannot start unless the previous one completes
				self.chk_process = None

				# `overwrite` supersedes if False
				if self.overwrite is False and self.keep_epoch_chk is False:
						self.keep_epoch_chk = True

				# Global ID of checkpoints being written
				# Used to format the checkpoint path
				# Instantiate from chk when restoring
				self.chk_global_id = -1

				# Sorted List of available checkpoints (fnames)
				self.available_chk_iters = self.mp_manager.list()
				self.available_chk_epochs = self.mp_manager.list()
				self.initalize_chk_dir()

				self.logger.info("Available checkpoints : ")
				for item in self.available_chk_iters:
					self.logger.info(item)
					

		"""
		The API to be used by the training code to initiate
		checkpoint.
		`additional_snapshot` : The iter, epoch, arch, and DL state must 
		be passed as a map if required.

		File is saved with the `global_id` suffix
		Only checkpoints at epoch boundaries are suffixed with epoch# instead of ID

		`is_epoch` : True if this is epoch boundary
		"""
		def save(
			self, \
			additional_snapshot=None, \
			is_epoch=False, \
			epoch=0, \
			synchronous=False, \
			profile_full=False,
			profile_snap=False,
			use_thread=False,
			persist=False):

				s = time.time()
				self.logger.info("[{}] ENTER SAVE FN".format(time.time()))
				self.chk_global_id += 1
				chk_fname = self.chk_prefix + str(self.chk_global_id)
				filepath = self._get_full_path(chk_fname)
				if is_epoch:
						chk_fname_link = self.chk_prefix + str(self.chk_global_id) + '_' +str(epoch) 
						filepath_link = self._get_full_path(chk_fname, epoch=True)

				self.logger.info("Writing chk {} at {}".format(self.chk_global_id, filepath))


				if synchronous:
					chk_fname_sync = chk_fname + '_sync'
					filepath_sync = self._get_full_path(chk_fname_sync)
					if not is_epoch:
						self.chk._serialize_and_persist_direct(
							self.chk.latest_snapshot, 
							filepath_sync, 
							additional_state=additional_snapshot, 
							persist=persist,
							iter_chk = self.available_chk_iters,
							overwrite = self.overwrite)
					else:
						chk_fname_link_sync = chk_fname_link + '_sync'
						filepath_link_sync = self._get_full_path(chk_fname_link_sync, epoch=True)
						self.chk._serialize_and_persist_direct(
							self.chk.latest_snapshot, 
							filepath_sync, 
							additional_state=additional_snapshot, 
							persist=persist,
							iter_chk = self.available_chk_iters,
							overwrite = self.overwrite,
							linkpath=filepath_link_sync,
							epoch_chk = self.available_chk_epochs)
					return


				# Check if there's an ongoing checkpoint operation
				if self.chk_process is not None:
						# There is an checkpoint underway. Wait
						if self.chk_process.is_alive():
								self.chk_process.join()

				# Once complete, initiate the next checkpoint synchronously
				self.logger.info("[{}] START SNAPSHOT".format(time.time()))
				success = self.chk._snapshot(self.active_snapshot.value, additional_state=additional_snapshot)

				if success:
					with self.lock:
						self.active_snapshot.value = 1

				dur_snap = time.time() -s

				if profile_snap:
					return dur_snap, 0

				if use_thread:
					fn = getattr(threading, 'Thread')
				else:
					fn = globals()["Process"]	
				print("Function is {}".format(fn))


				# Start persist asynchronously
				if not is_epoch:
					keywords = { \
						'iter_chk':self.available_chk_iters, \
						'overwrite':self.overwrite}
					self.chk_process = \
						fn(target=self.chk._serialize_and_persist,	\
						args=[filepath, self.chk.latest_snapshot, self.active_snapshot, self.lock], kwargs=keywords)
				else:
					keywords = { \
						'iter_chk':self.available_chk_iters, \
						'overwrite':self.overwrite, \
						'epoch_chk':self.available_chk_epochs,\
						'linkpath': filepath_link}
					self.chk_process = \
						fn(target=self.chk._serialize_and_persist,\
						args=[filepath, self.chk.latest_snapshot, self.active_snapshot, self.lock], kwargs=keywords)

				self.logger.info("[{}] CALL PROCESS NOW".format(time.time()))
				self.chk_process.start()
				self.logger.info("[{}] RETURN FROM START".format(time.time()))
				
				if profile_full:
						self.chk_process.join()

				dur = time.time() -s

				
				#time.sleep(1)
				return dur_snap, dur 



		def save_cpu(
			self, \
			additional_snapshot=None, \
			is_epoch=False, \
			epoch=0, \
			synchronous=False, \
			persist=False,
			profile_snap=False,
			profile_full=False,
			use_thread=True):
				self.logger.info("[{}] ENTER SAVE FN".format(time.time()))

				s = time.time()
				self.chk_global_id += 1
				chk_fname = self.chk_prefix + str(self.chk_global_id)
				filepath = self._get_full_path(chk_fname)
				if is_epoch:
						chk_fname_link = self.chk_prefix + str(self.chk_global_id) + '_' +str(epoch) 
						filepath_link = self._get_full_path(chk_fname, epoch=True)

				self.logger.info("Writing chk {} at {}".format(self.chk_global_id, filepath))

				# Check if there's an ongoing checkpoint operation
				if self.chk_process is not None:
						# There is an checkpoint underway. Wait
						if self.chk_process.is_alive():
								self.chk_process.join()

				self.logger.info("Starting next snapshot {:.2f}s".format(time.time()-s))

				# Once complete, initiate the next checkpoint synchronously
				self.logger.info("[{}] SAVE FN SNAP NOW".format(time.time()))

				snap_ptr = {}
				for name, ref in self.chk.tracking_map.items():
					snap_ptr[name] = ref.state_dict()


				# check current snapshot status
				if self.active_snapshot.value == 1:
					self.logger.info("ERROR! Active snapshot")
					return
	
				with self.lock:
					self.in_progress_snapshot.value = 1

				self.logger.info("[{}] START SAVE CALL".format(time.time()))

				if synchronous:
					self.chk._snapshot_and_persist_async(filepath, self.active_snapshot, self.in_progress_snapshot, self.lock, snap_ptr, iter_chk=self.available_chk_iters, overwrite=self.overwrite)
					self.logger.info("Returned from save in {:.2f}s".format(time.time()-s))
					self.logger.info("[{}] END SAVE".format(time.time()))
					return 
				
				if use_thread:
					fn = getattr(threading, 'Thread')
				else:
					fn = globals()["Process"]	
				print("Function is {}".format(fn))
				if not is_epoch:
					keywords = { \
						'iter_chk':self.available_chk_iters, \
						'overwrite':self.overwrite, \
						'profile': profile_snap }
					self.chk_process = \
						fn(target=self.chk._snapshot_and_persist_async,	\
						args=[filepath, self.active_snapshot, self.in_progress_snapshot, self.lock, snap_ptr], kwargs=keywords)
				else:
					keywords = { \
						'iter_chk':self.available_chk_iters, \
						'overwrite':self.overwrite, \
						'epoch_chk':self.available_chk_epochs,\
						'linkpath': filepath_link, \
						'profile': profile_snap }
					self.chk_process = \
						fn(target=self.chk._snapshot_and_persist_async,\
						args=[filepath, self.active_snapshot, self.in_progress_snapshot, self.lock, snap_ptr], kwargs=keywords)

				self.chk_process.start()
		
				if profile_snap or profile_full:
						self.chk_process.join()

				dur = time.time() -s
				#time.sleep(1)
				self.logger.info("Returned from save in {:.2f}s".format(time.time()-s))
				self.logger.info("[{}] END SAVE".format(time.time()))
				return 0, dur 



		
		"""
		Restores the latest checkpoint among all available, or the latest
				epoch boundary checkpoint corresponding to `epoch`

		Returns : Map of state of items that were not resumed yet
				This should be same as the map passed in to the save() 
				call by the DL/script. 
				These restorations are assumed to happen in the script/DL
				If nothing remains to be restore, returns None
		"""
		def restore(self, latest=True, epoch=0, gpu=0):
				fname = self.get_latest_checkpoint(latest=latest, epoch=epoch)
				if fname is None:
					return None
				filepath = self._get_full_path(fname, epoch=not latest)
				self.logger.info("Latest checkpoint is {}".format(filepath))
				extra_state = self.chk._restore(filepath=filepath, gpu=gpu)
				return extra_state


		def initalize_chk_dir(self):
				if os.path.exists(self.chk_dir):
						# Get list of all files
						chk_files = [os.path.splitext(f)[0] for f in os.listdir(self.chk_dir) if isfile(os.path.join(self.chk_dir, f))]
						self.logger.info(chk_files)
						chk_files.sort(key=natural_keys)

						for files in chk_files:
								self.available_chk_iters.append(files)
						del chk_files

						epoch_chk_dir = os.path.join(self.chk_dir, self.chk_epoch_subdir)
						if os.path.exists(epoch_chk_dir):
								epoch_chk_files = [os.path.split(f)[0] for f in os.listdir(epoch_chk_dir) if isfile(os.path.join(epoch_chk_dir, f))]
								epoch_chk_files.sort(key=natural_keys)
								for files in epoch_chk_files:
										self.available_chk_epochs.append(files)
								del epoch_chk_files
						else:
								os.makedirs(epoch_chk_dir)
				else:
						os.makedirs(self.chk_dir)

				
		def get_latest_checkpoint(self, latest=True, epoch=0):
			"""
			Returns the full path of the latest checkpoint
				`latest` : If true, return most recent checkpoint
						If false, return chk corresponding to `epoch` 
						if available
			"""
			fname = None
			if latest and len(self.available_chk_iters) > 0:
				fname = self.available_chk_iters[-1]
			elif len(self.available_chk_epochs) > 0:
				fname = self.available_chk_epochs[-1]

			return fname


		def _get_full_path(self, fname, epoch=False):
				if not epoch:
						return os.path.join(self.chk_dir, fname + '.chk')
				else:
						return os.path.join(self.chk_dir, self.chk_epoch_subdir, fname + '.chk')
	
		def weight_update(self):
				if 'optimizer' in self.chk.tracking_map:
						optimizer = self.chk.tracking_map['optimizer']
						s = time.time()
						while self.in_progress_snapshot.value == 1:
							continue
#							self.logger.info("Progresssss")
						optimizer.step()
						#torch.cuda.synchronize()
						dur = time.time() - s
						#self.logger.info("Stall to weight update = {}s".format(dur))
				else:	
						self.logger.info("NO Optimizer found")

		# Returns size of tensors of all tractable items in MB
		@ property
		def get_chk_size(self):
			snap_ptr = {}
			size = 0
			for name, ref in self.chk.tracking_map.items():
				snap_ptr[name] = ref.state_dict()
				size += _get_all_size(snap_ptr[name])
			return size/1024/1024

def _get_all_size(ele, sz = 0):
		if torch.is_tensor(ele):
			sz += ele.nelement()*ele.element_size()
		elif isinstance(ele, dict):
			for k,v in ele.items():
				sz = _get_all_size(v, sz)
		elif isinstance(ele, list):
			for v in ele:
				sz = _get_all_size(v, sz)
		else:
				sz += sys.getsizeof(ele)

		return sz

def _save(filepath, snap):
		torch.save(snap,filepath)

def atoi(text):
		return int(text) if text.isdigit() else text


def natural_keys(text):
		return [ atoi(c) for c in re.split(r'(\d+)', text) ]
