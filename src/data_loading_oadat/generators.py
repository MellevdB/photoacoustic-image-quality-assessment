# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import numpy as np
import h5py
import tensorflow as tf

class Generator_Group_Sampler(tf.keras.utils.Sequence):
    '''A generator that will sample uniformly across list of generators until ALL generators are exhausted.'''
    def __init__(self, l_gens):
        self.l_gens = l_gens
        self.l_iter = None
        self.len = len(self)
        self.inds = None
        self.__generate_gen_lookup()
        if hasattr(self.l_gens[0], 'prng'):
            self.prng = self.l_gens[0].prng
        if hasattr(self.l_gens[0], 'shuffle'):
            self.shuffle = self.l_gens[0].shuffle

    def __generate_gen_lookup(self):
        for i, gen in enumerate(self.l_gens):
            if i == 0:
                inds = np.asarray(gen.inds)
                gen_inds = np.ones(len(gen.inds)) * i
            else:
                inds_ = np.asarray(gen.inds)
                gen_inds_ = np.ones(len(gen.inds)) * i
                inds = np.concatenate((inds, inds_), axis=0)
                gen_inds = np.concatenate((gen_inds, gen_inds_), axis=0)
        self.inds = list(zip(gen_inds, inds))
    
    def __len__(self):
        '''Generator expects that the passed generators all implement __len__'''
        return np.sum(len(gen) for gen in self.l_gens)

    def __iter__(self):
        '''Generator expects that the passed generators all implement __iter__'''
        self.l_iter = [iter(gen) for gen in self.l_gens]
        len_inds = [len(gen) for gen in self.l_gens]
        len_inds_max = np.max(len_inds)
        for i in range(len_inds_max):
            for i_gen, it in enumerate(self.l_iter):
                if i < len_inds[i_gen]:
                    s = next(it)
                    yield s
    def __getitem__(self, index):
        gen_ind, ind_in_gen = self.inds[index]
        s = self.l_gens[gen_ind][ind_in_gen]
        return s    

class Generator_Paired_Input_Output:
    '''Samples (in_key, out_key) sample pairs from fname_h5 and applies transforms on in_key and transforms_target on out_key'''
    def __init__(self, fname_h5, in_key, out_key, inds=None, transforms=None, transforms_target=None, **kwargs):
        self.fname_h5 = fname_h5
        self.inds = inds
        self.in_key = in_key
        self.out_key = out_key
        self.transforms = transforms
        self.transforms_target = transforms_target
        self.prng = kwargs.get('prng', np.random.RandomState(42))
        self.shuffle = kwargs.get('shuffle', False)
        self.len = None #will be overwritten in check_data()
        self.check_data()

    def check_data(self,):
        len_ = None
        with h5py.File(self.fname_h5, 'r') as fh:
            for k in [self.in_key, self.out_key]:
                if len_ is None: 
                    len_ = fh[k].shape[0]
                if len_ != fh[k].shape[0]:
                    raise AssertionError('Length of datasets vary across keys. %d vs %d' % (len_, fh[k].shape[0]))
        if self.inds is None:
            self.len = len_
            self.inds = np.arange(len_)
        else:
            self.len = len(self.inds)

    def __len__(self):
        return self.len

    def __getitem__(self, index): 
        with h5py.File(self.fname_h5, 'r') as fh:
            x = fh[self.in_key][index,...]
            if self.transforms is not None:
                x = self.transforms(x)

            y = fh[self.out_key][index,...]
            if self.transforms_target is not None:
                y = self.transforms_target(y)
        return (x,y)

    def __iter__(self):
        inds = np.copy(self.inds)
        if self.shuffle:
            self.prng.shuffle(inds)
        for i in inds:
            s = self.__getitem__(index=i)
            yield s

class Generator_Paired_Input_Output_Seq(tf.keras.utils.Sequence):
    def __init__(self, batch_size, fname_h5, in_key, out_key, inds=None, transforms=None, transforms_target=None, **kwargs):
        self.gen = Generator_Paired_Input_Output(fname_h5, in_key, out_key, inds, transforms, transforms_target, **kwargs)
        self.batch_size = batch_size
        self.fname_h5 = fname_h5
        self.inds = self.gen.inds.copy()
        self.in_key = in_key
        self.out_key = out_key
        self.transforms = transforms
        self.transforms_target = transforms_target
        self.prng = self.gen.prng
        self.shuffle = self.gen.shuffle
        self.len = int(self.gen.len // self.batch_size)
        self.on_epoch_end()
        self._x_b, self._y_b = [], []

    def __getitem__(self, index):
        '''Returns a complete batch'''
        inds = self.inds[index*self.batch_size:(index+1)*self.batch_size]
        batch = []
        for ind in inds:
            sample = self.gen[ind]
            if sample is None:
                print(f'Warning, queried sample at index: {ind} returned None.')
                continue
            self._add_sample_to_batch(sample)
        batch = self._compile_batch()
        return batch

    def __len__(self,):
        return self.len

    def __iter__(self):
        inds = self.inds
        for i_iter, ind in enumerate(inds): ## note that an epoch will not go over whole dataset, but just go over one patch for each time point in the dataset.
            sample = self.gen[ind]
            if sample is None:
                continue
            self._add_sample_to_batch(sample)
            if len(self._x_b) == self.batch_size:
                batch = self._compile_batch()
                yield batch

    def _add_sample_to_batch(self, sample):
        x,y = sample
        self._x_b.append(x)
        self._y_b.append(y)
        
    def _compile_batch(self, drop_y2=True):
        x_b = np.asarray(self._x_b)
        y_b = np.asarray(self._y_b)
        batch = (x_b, y_b)
        self._x_b, self._y_b = [], []
        return batch
    
    def on_epoch_end(self,):
        if self.gen.shuffle:
            inds = self.inds
            self.gen.prng.shuffle(inds)
    

    