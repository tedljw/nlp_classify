import fire
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main(cmd):
    if cmd == 'train' :
        return "finish model train"

    if cmd == 'predict' :
        return "finish model predict"

    if cmd == 'distill' :
        return "finish model distill"

if __name__ == '__main__':
    fire.Fire(main)



