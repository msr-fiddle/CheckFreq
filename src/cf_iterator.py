
import torch
import time
from collections.abc import Iterable
import math
from statistics import mean
from cf_manager import CFMode
from disk_bw import get_storage_bandwidth
import json
import os

# DALI dataloader imports
try:
	from nvidia.dali.plugin.pytorch import DALIClassificationIterator
except:
	raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")  

#Pytorch native dataloader import
from torch.utils.data.dataloader import DataLoader


class CFIterator:
	def __init__(
			self, 
			dataloader, 
			bs=128, 
			dali=True, 
			steps_this_epoch=0, 
			epoch=0, 
			arch='resnet18',
			worker_id=0,
			chk_freq=0, 
			steps_to_run=-1,
			max_overhead=5,
			dynamic = False,
			persist=True,
			cf_manager=None):
		if not isinstance(dataloader, Iterable):
			raise ValueError("Dataloader of type {} is not iterable".format(type(dataloader)))

		self._dataloader = dataloader
		print("Using CF Iterator")
		self._steps_this_epoch = steps_this_epoch
		self._samples_this_epoch = 0
		self._epoch = epoch
		self._chk_freq = chk_freq
		self._batch_size = bs
		self._dali = dali
		self._persist = persist
		self._steps_to_run = steps_to_run
		self._max_steps_per_epoch = int(math.ceil(self._size / self._batch_size))
		self._total_steps =  self._epoch*self._max_steps_per_epoch + self._steps_this_epoch
		self._worker_id = worker_id
		self._steps_this_run = 0
		self._exit = False
		self._cf_manager = cf_manager
		self._chk_fn = None
		self._start_monitor = False
		self._stop_monitor = False
		self._max_overhead = max_overhead
		self._arch = arch
		self._cache_file = ".cache_" + self._arch + "_" + str(self._batch_size)
		self._steps_since_chk = 0
		self.t_start = 0
		self._dynamic = dynamic
		# default MANUAL determine frequency
		self._chk_mode = CFMode.MANUAL

		# Have we already profiled for ideal chk freq
		self._profile_done = False

		if self._cf_manager is not None:
			self._chk_mode = self._cf_manager.mode
		
		if self._worker_id == 0:
			if self._chk_mode == CFMode.MANUAL and self._chk_freq == 0:
				print("No iter level checkpointing chosen in MANUAL mode")
			elif self._chk_mode == CFMode.MANUAL and self._chk_freq > 0:
				print("Freq {} chosen in MANUAL mode".format(self._chk_freq))


		self._profile_iter_count = 0
		self._prev_iter_end = time.time()
		self._iter_dur = []
		self._mem_util = []
		self._avg_iter_dur = 0
		self._monitor_iter_count = -1
		self._skip_few_monitors = False
		#self._use_thread = True
		self._use_thread = False

		# If AUTO mode, fetch values from cache if present
		if self._worker_id == 0:
			if self._chk_mode == CFMode.AUTO:
				if os.path.exists(self._cache_file):
					self.load_params_from_cache()

		print("Epoch={}, this ep steps={}, max_steps={}, total_steps={}, steps_to_run = {}".format(self._epoch,
			self._steps_this_epoch, self._max_steps_per_epoch, self._total_steps, self._steps_to_run))

	def load_params_from_cache(self):
		found_all = True
		with open(self._cache_file) as f:
			data = json.load(f)
			for key, val in data.items():
				if key == 'avg_iter_dur':
					self._avg_iter_dur = val
				elif key == 'chk_freq':
					self._chk_freq = val
				elif key == 'self._chk_fn':
					self.populate_chk_fn(val)
				elif key == 'self._use_thread':
					self._use_thread = val
				else:
					found_all = False

		if found_all:
			self._profile_done = True
			# To skip the first 5 iters for profiling
			self._skip_few_monitors = True
			self._monitor_iter_count = 0
			print("Loaded : iter_dur = {}s, freq={}, fn={}, th={}".format(self._avg_iter_dur,self._chk_freq,self._chk_fn, self._use_thread))
				

	def cache_params(self):
		data = {
			'avg_iter_dur' : self._avg_iter_dur,
			'chk_freq' : self._chk_freq,
			'self._chk_fn' : self.get_chk_fn_str(),
			'self._use_thread' : self._use_thread
		}
		print("CACHING : ")
		print(data)
		with open(self._cache_file, 'w') as f:
			json.dump(data, f)
			os.fsync(f.fileno())
			
	def get_chk_fn_str(self):
		if self._chk_fn == getattr(self._cf_manager, "save_cpu"):
			return "cpu"
		return "gpu"	

	def populate_chk_fn(self, dev):
		if dev == "cpu":
			self._chk_fn = getattr(self._cf_manager, "save_cpu")
		else:
			self._chk_fn = getattr(self._cf_manager, "save")
		

		
	def __iter__(self):
		self._iterator = iter(self._dataloader)
		return self

	def __next__(self):

		# Profile if required on main worker
		if self._worker_id == 0 and self._chk_mode == CFMode.AUTO and not self._profile_done:
				self._profile_iter_count += 1
				dev = max(0, torch.cuda.current_device())
				if self._profile_iter_count < 100 and self._profile_iter_count >= 5:
						print("PROFILE step {}".format(self._profile_iter_count))	
						self._iter_dur.append(time.time()-self._prev_iter_end)
						# Change to profile
						self._mem_util.append(torch.cuda.max_memory_allocated(dev))
				elif self._profile_iter_count == 100:
						self._complete_profile(dev)
						
		# Checkpoint if required on main worker
		elif self._worker_id == 0 and self._chk_freq > 0 and self._steps_since_chk == self._chk_freq:
		#elif self._worker_id == 0 and self._chk_freq > 0 and self._total_steps % self._chk_freq == 0 and self._steps_since_chk == self._chk_freq:
			print("MUST CHECKPOINT NOW AT ITER {}, steps {}".format(self._steps_this_epoch, self._steps_since_chk))
			if self._chk_mode == CFMode.MANUAL:
				self._cf_manager.save(synchronous=True, additional_snapshot=self.state_dict(), persist=self._persist)
				#self._cf_manager.save(synchronous=True, additional_snapshot=self.state_dict(), persist=False)
			else:
				# Iter-level chk
				self._chk_fn(additional_snapshot=self.state_dict(), use_thread=self._use_thread) 
			
			self._steps_since_chk = 0    
 
			if self._profile_done and not self._start_monitor and self._chk_mode == CFMode.AUTO and not self._stop_monitor:
				self._iter_dur = []
				self._start_monitor = True


		if self._worker_id == 0 and self._start_monitor and not self._stop_monitor:
			self._monitor_iter_count += 1
				
			if self._monitor_iter_count == 1:
				self.t_start = time.time()
			elif self._monitor_iter_count == self._chk_freq + 1:
				t_dur = time.time() - self.t_start
				print("Duration between checkpoints = {:.2f}s".format(t_dur))

			if self._monitor_iter_count > 0 and self._monitor_iter_count <= self._chk_freq:
				self._iter_dur.append(time.time()-self._prev_iter_end)
			elif self._monitor_iter_count == self._chk_freq + 1:
				if self._skip_few_monitors and len(self._iter_dur) > 3:
					self._iter_dur = self._iter_dur[2:-1]
					self._skip_few_monitors = False
	
				current_iter_mean =  mean(self._iter_dur)
				current_total = sum(self._iter_dur)
				orig_total =  self._avg_iter_dur*len(self._iter_dur)

				# Update avg iter dur
				num_items = len(self._iter_dur) - int(len(self._iter_dur)/3)
				if num_items > 3:
					new_avg = mean(self._iter_dur[num_items:])
					print("New avg = {:.3f}s".format(new_avg))
				

				overhead = max(0, current_iter_mean - self._avg_iter_dur)
				overhead_full = current_total - orig_total
				overhead_percent = overhead_full/orig_total*100
				print("NEW OVERHEAD IS  ={:.2f}, FULL={:.2f}".format(overhead, overhead_full))
				print(self._iter_dur)
				print("OLD ITER={:.2f}, NEW_ITER={:.2f}".format(self._avg_iter_dur, current_iter_mean))
				print("OLD TOTAL={:.2f}, NEW_TOTAL={:.2f}, percent={:.2f}%".format(orig_total,  current_total, overhead_percent))
				if overhead_percent > self._max_overhead:
						self._chk_freq += 2
						self.cache_params()
						print("Changed chk freq to {}".format(self._chk_freq))
				elif not self._dynamic:
						self._start_monitor = False
						self._stop_monitor = True

				self._iter_dur = []
				self._monitor_iter_count = 0

				#self._start_monitor = False


		# If we reached the max specified steps, return
		# To simulate a force stop (crash/interruption every n iters)
		if self._steps_to_run >= 0:
			if self._steps_this_run == self._steps_to_run:
				#print("MUST CHECKPOINT NOW AT ITER {}:{}:{}".format(self._epoch, self._steps_this_epoch, self._samples_this_epoch))
				self._exit = True
				print("Epoch={}, this ep steps={}, total_steps={}, steps_this_run = {}".format(self._epoch, self._steps_this_epoch, self._total_steps, self._steps_this_run))	
				raise StopIteration


		self._prev_iter_end  = time.time()
		
		#Else get next batch from iterator
		try:
			val = next(self._dataloader)
		except StopIteration:
			print("Tracking epoch end in CF DL. Steps this run = {}, this epoch={}, samples this epoch={}".format(self._steps_this_run, self._steps_this_epoch, self._samples_this_epoch))
			self._epoch += 1
			self._steps_this_epoch = 0     
			self._samples_this_epoch = 0     
			self._steps_since_chk = 0     
			print("Epoch set to {}".format(self._epoch))
			
			# Reached epoch boundary. Force a chk
			if self._worker_id == 0 and self._chk_mode == CFMode.MANUAL:
				self._cf_manager.save(
					synchronous=True, 
					additional_snapshot=self.state_dict(), 
					persist=self._persist,
					is_epoch = True,
					epoch = self._epoch)
			elif self._worker_id == 0:
				self._chk_fn(
					additional_snapshot=self.state_dict(),
					is_epoch = True,
					epoch = self._epoch) 

			raise StopIteration

		self._total_steps += 1
		self._steps_this_epoch += 1
		self._samples_this_epoch += self._batch_size
		self._steps_this_run += 1
		self._steps_since_chk += 1

		return val

		

	def _complete_profile(self, dev):
		self._avg_iter_dur = mean(self._iter_dur)
		self._avg_free_mem = (torch.cuda.get_device_properties(dev).total_memory - mean(self._mem_util))/1024/1024
		print("Average iter dur = {:.3f}, free mem = {:.2f}MB".format(self._avg_iter_dur, self._avg_free_mem))

		# Get checkpoint size in MB
		# Ignore DL size as its minimal
		self._chk_size = self._cf_manager.get_chk_size
		print("Size of chk = {:.3f}MB".format(self._chk_size))

		disk_path = self._cf_manager.chk_dir
		if disk_path.startswith("./"):
			disk_path = os. getcwd()
		print("Disk path = {}".format(disk_path))
		disk_bw = get_storage_bandwidth(disk_path)
		time_to_disk = self._chk_size/515.0
		print("Time to persist = {:.3f}s".format(time_to_disk))
		# decide the snapshot type
		t_g = t_s = t_c = t_f = 0

		t_w = t_i = self._avg_iter_dur

    # Profile only the snapshot phase - CPU based
		_, t_ct = self._cf_manager.save_cpu(profile_snap=True, use_thread=True)
		_, t_cp = self._cf_manager.save_cpu(profile_snap=True, use_thread=False)
		if t_ct <= t_cp:
			t_c = t_ct
			self._use_thread = True
		else:
			t_c = t_cp
			self._use_thread = False
		print("t_cp={:.4f}, t_ct={:.4f}".format(t_cp, t_ct))

		# By default, set CPU based snapshot
		self._chk_fn = getattr(self._cf_manager, "save_cpu") 
		
		# Although we use threading, due to Python imperfections we might
		# see as much as t_c overhead in one iter
		overhead = t_c
		#overhead = max(0, t_c - t_w)

		# If mem available, and low overhead, 
		# do synchronous GPU snapshot
		if self._chk_size < self._avg_free_mem:
			# Use multi-proc by default
			t_gt, t_ft = self._cf_manager.save(profile_full=True, use_thread=True)
			t_gp, t_fp = self._cf_manager.save(profile_full=True, use_thread=False)

			if t_fp <= t_ft:
					t_f = t_fp
					t_g = t_gp
					self._use_thread = False
			else:
					t_f = t_ft
					t_g = t_gt
					self._use_thread = True


			print("t_fp={:.4f}, t_ft={:.4f}, t_gp={:.4f}, t_gt={:.4f}".format(t_fp, t_ft, t_gp, t_gt))
			t_s = t_f - t_g

			if overhead >= t_g:
					# Use GPU-snapshot
					self._chk_fn = getattr(self._cf_manager, "save")
					#self._use_thread = True
					#self._use_thread = False
					overhead = t_g

		print("t_c={:.4f}, t_g={:.4f}, t_s={:.4f}, t_w={:.4f}".format(t_c, t_g, t_s, t_w))
		print("Chosen function is : {}, overhead={}".format(self._chk_fn, overhead))

		_, t_f = self._chk_fn(profile_full=True, use_thread=self._use_thread)
		
		self._chk_freq = max(math.ceil((t_f - overhead)/t_i), 1)

		percent_overhead = overhead/self._chk_freq*t_i*100

		self.cache_params()

		self._profile_done = True
		self._steps_since_chk = 0

		print("Chosen freq = {}, percent_ov={:.2f}%".format(self._chk_freq, percent_overhead))


	def state_dict(self):
		state = {
			'iter' : self._steps_this_epoch,
			'steps_so_far' : self._total_steps, 
			'start_index' : self._samples_this_epoch,
			'epoch' : self._epoch}
		return state

	def load_state_dict(self, chk_map):
		if chk_map is None:
			return 

		self._steps_this_epoch = chk_map['iter']
		self._total_steps = chk_map['steps_so_far']
		self._samples_this_epoch = chk_map['start_index']
		self._epoch = chk_map['epoch']


	def profile_all(self):
		snap_thread_time = self._cf_manager.save_cpu(profile_snap=True, use_thread=True)
		snap_proc_time = self._cf_manager.save_cpu(profile_snap=True, use_thread=False)
		full_thread_time = self._cf_manager.save_cpu(profile_full=True, use_thread=True)
		full_proc_time = self._cf_manager.save_cpu(profile_full=True, use_thread=False)
		dur_snap_1, full_proc_gpu_time = self._cf_manager.save(profile=True, use_thread=False)
		dur_snap_2, full_th_gpu_time = self._cf_manager.save(profile=True, use_thread=True)
		print("Snap th={:.2f}\nSnap proc={:.2f}\nFull Th={:.2f}\nFull proc={:.2f}\nFull GPU proc={:.2f}\nFull GPU th={:.2f}\nGPU snap 1= {:.2f}\nGPU snap 2={:.2f}".format(snap_thread_time,snap_proc_time,full_thread_time,full_proc_time, full_proc_gpu_time, full_th_gpu_time, dur_snap_1, dur_snap_2))

		


	def reset(self):
		return self._dataloader.reset()


	@property
	def exit(self):
		return self._exit	

	@property
	def _size(self):
		return self._dataloader._size
		

