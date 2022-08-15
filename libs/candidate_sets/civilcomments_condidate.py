import json
from multiprocessing import Process

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from libs.utils.wilds_utils import WILDS_utils

import numpy as np
from nltk.corpus import wordnet  # Import wordnet from the NLTK

import spacy
nlp = spacy.load('en_core_web_lg')

dataset_name = "civilcomments"
dataset = WILDS_utils(dataset_name).dataset
import tqdm





class CivilComments_Candidate_Set:
    def __init__(self, reshape_size, batch_size):
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)

    def get_loader_dict(self):
        return {
            'orig':self.get_train_loader_orig,
            'mask': self.get_train_loader_mask,
        }

    def get_train_dataloader(self, batch_size, transform):
        dataset_ = dataset.get_subset(split="train")
        return dataset_, DataLoader(dataset_, batch_size=batch_size,num_workers=12,
                                    shuffle=False)

    def get_train_loader_orig(self):
        print("1")
        traindata_orig = dataset.get_subset(split="train")
        print("1")
        trainloader_orig = DataLoader(traindata_orig, batch_size=self.batch_size,num_workers=12,
                                      shuffle=False)
        print("1")
        return trainloader_orig

    def get_train_loader_mask(self):
        print("2")
        traindata_orig = self.civil_edit_dataset('mask','train')
        print("2")
        trainloader_orig = DataLoader(traindata_orig, batch_size=self.batch_size,num_workers=12,
                                      shuffle=False)
        print("2")
        return trainloader_orig


    def get_test_loader(self):
        test_data = dataset.get_subset(
            "test",
        )
        testloader = DataLoader(test_data, batch_size=self.batch_size,
                                shuffle=False)
        return testloader

    def get_val_loader(self):
        val_data = dataset.get_subset(
            "val",
        )
        valloader = DataLoader(val_data, batch_size=self.batch_size,num_workers=12,
                               shuffle=False)
        return valloader

    def get_metadata(self, loader):
        metadata_all = []
        for _, (_, _, metadata) in enumerate(loader):
            metadata_all.append(metadata)
        metadata_all = np.vstack(metadata_all)
        return metadata_all

    def get_train_metadata(self):
        trainloader_orig = self.get_train_loader_orig()
        train_metadata = self.get_metadata(trainloader_orig)
        return train_metadata

    def get_test_metadata(self):
        testloader = self.get_test_loader()
        test_metadata = self.get_metadata(testloader)
        return test_metadata

    def get_val_metadata(self):
        valloader = self.get_val_loader()
        val_metadata = self.get_metadata(valloader)
        return val_metadata

    class civil_edit_dataset(Dataset):
        def __init__(self,task,split):
            # try to load processed nlp file
            dict_file_name = 'dict.json'

            self.orig_dataset = dataset.get_subset(split=split)
            self.id_list = ['male','female','transgender','gender','heterosexual','gay','lesbian','bisexual','sexual orientation',
                       'christian','jewish','muslim','hindu','buddhist','atheist','religion','black','white','asian',
                       'latino','race','ethnicity','physical disability','intellectual', 'learning disability','psychiatric illness',
                       'mental illness','LGBTQ','asian','latino','orientation','medicine','woman','man','black']
            # need to remove labeled words based on the metadata
            self.synonyms = ['medicine','woman','man','black','blacks','men','women']
            for words in self.id_list:
                synset = wordnet.synsets(words)


                for l in synset:
                    lemmas = l.lemmas()
                    for word in lemmas:
                        word = word.name()
                        #print(word)
                        if word not in self.synonyms:
                            self.synonyms.append(word)
            self.spacy_synonyms = []
            for each in self.synonyms:
                self.spacy_synonyms.append(nlp(each.lower()))

            self.threshold = 0.7
            self.count = 0


            self.dataset = dict()
            i = 0
            for _ in tqdm.tqdm(self.orig_dataset):
                #p = Process(target=self.remove_words,args=(i,))
                self.remove_words(i)
                #p.start()
                #p.join()
                i+=1


            self.combined_dataset = dict()
            for idx in range(len(self.dataset)):
                self.combined_dataset.update({
                    str(idx):{
                        'orig':self.orig_dataset[idx][0],
                        'new_': self.dataset[str(idx)][0]
                    }
                })




            print(len(self.orig_dataset))
            print(len(self.dataset))

            with open(dict_file_name,'w') as f:
                json.dump(self.combined_dataset,f)
            print(self.count)
            print("Done with process data")
            #exit(-1)

        def remove_words(self, idx):
            data = self.orig_dataset[idx]
            sentence = data[0]
            label = data[1]
            matadata = data[2]
            #print(idx)


            # doc = nlp(sentence)
            # token_list = []
            # for token in doc:
            #     add = True
            #     # print('token: ' + token.text.lower())
            #     #g1 = nlp(token.text.lower())
            #     for each in self.synonyms:
            #         if token.text in ['.',',','!']: break
            #         #each = nlp(each)
            #         #print(each)
            #         #if (g1 and g1.vector_norm) and (each and each.vector_norm):
            #         if True:
            #             #if g1.similarity(each)>= self.threshold:
            #             if each == token.text:
            #                 #print(token,each.text)
            #                 self.count += 1
            #                 add = False
            #                 break
            #     if add:
            #         token_list.append(token.text)
            #     else:
            #         token_list.append('None')
            added = False
            for each in self.synonyms:
                temp_word = ''+each+''

                if temp_word.lower() in sentence.lower():

                    sentence = sentence.lower().replace(temp_word.lower(),' None ')
                    added = True
            if added:
                self.count+=1


            #
            #
            #
            #
            #
            #
            # for each_syn in self.synonyms:
            #     each_syn = each_syn
            #     if each_syn in sentence:
            #         sentence = sentence.replace(each_syn,'None')
            # print(sentence)
            #sentence = ' '.join(each for each in token_list)
            # print(sentence)
            # exit()
            self.dataset.update({str(idx):[sentence,label,matadata]})
            return
            #self.dataset.append([sentence,label,matadata])


        def __len__(self):
            return len(self.orig_dataset)
        def __getitem__(self,idx):
            #return self.remove_words(idx)
            return self.dataset[str(idx)]