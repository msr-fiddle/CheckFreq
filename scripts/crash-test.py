import torch
import os
import sys
import argparse
import time

parser = argparse.ArgumentParser("Test torch.save")
parser.add_argument('--persist', action='store_true', default=False)
parser.add_argument('--one', action='store_true', default=False)
parser.add_argument('--check', action='store_true', default=False)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--sleep', action='store_true', default=False)
parser.add_argument('--num_tensors', type=int, default=3)
parser.add_argument('--dir', type=str, default='./chk/')

torch.manual_seed(2)

args=parser.parse_args()

def main():

  if args.check:
    check_results()
    return

  tensor_bank={}
  tensor_bank_ow={}
  for i in range(args.num_tensors):
    tensor_bank[i] = torch.randn(256,3,224,224)
    tensor_bank_ow[i] = torch.randn(256,3,224,224)

  if not os.path.exists(args.dir):
    os.makedirs(args.dir)

  filepath = os.path.join(args.dir, 'model.chk')

  print("Starting save to {}..".format(filepath))
  s = time.time()
  torch.save(tensor_bank, filepath)
  if args.persist:
    persist(filepath)

  if args.overwrite:
    filepath_ow = os.path.join(args.dir, 'model.chk')
  else:
    filepath_ow = os.path.join(args.dir, 'model_new.chk')

  if not args.one:
    torch.save(tensor_bank_ow, filepath_ow)
    if args.persist:
      persist(filepath_ow)

  dur = time.time() - s 
  print("Returned from save in {:.2f} s".format(dur))

  if args.sleep:
    time.sleep(30)

def persist(filepath):
  with open(filepath) as f:
    os.fsync(f.fileno())


def check_results():
  new_ten_1 = torch.load('chk/model.chk')
  old_ten_2 = torch.load('chk-compare/model_new.chk')
  old_ten_1 = torch.load('chk-compare/model.chk')


  if args.overwrite:
    for idx, val in new_ten_1.items():   
      print("Ten 1 : {}".format(torch.all(torch.eq(val, old_ten_1[idx]))))
      print("Ten 2 : {}".format(torch.all(torch.eq(val, old_ten_2[idx]))))
    return


  new_ten_2 = torch.load('chk/model_new.chk')

  for idx, val in new_ten_1.items():   
    print("Ten 1 : {}".format(torch.all(torch.eq(val, old_ten_1[idx]))))
  for idx, val in new_ten_2.items():   
    print("Ten 2 : {}".format(torch.all(torch.eq(val, old_ten_2[idx]))))

if __name__ == '__main__':
  main()
