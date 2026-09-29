"""Shared compute ceiling. Idle workers are not simultaneous image computations."""
import os

MAX_THREADS=32
THREADS=min(MAX_THREADS,max(1,os.cpu_count() or 1))

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]=str(THREADS)

def configure():
    import cv2
    cv2.setNumThreads(THREADS)

def session_options():
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.intra_op_num_threads=THREADS
    options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    return options
