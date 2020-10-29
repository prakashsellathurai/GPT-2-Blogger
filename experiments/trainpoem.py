import gpt_2_simple as gpt2
from datetime import datetime

gpt2.download_gpt2(model_name="355M")
file_name = "/data/test/2019-03-06-gpt2-poetry-1000samples.txt"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='355M',
              steps=5000,
              restore_from='latest',
              run_name='poem',
              print_every=10,
              sample_every=5000,
              )
