import os
import shutil

if __name__ == '__main__':
    import ray.rllib.models.torch.complex_input_net

    shutil.copy2(os.path.join('patches_for_rllib', 'complex_input_net.py'),
                 ray.rllib.models.torch.complex_input_net.__file__)

    print('Patched rllib')
