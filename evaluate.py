import argparse


def evaluate(generator, dataloader):
    generator.eval()
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
