import models
from datetime import datetime
from utils import saveHistory
from solver import VoiceTechniqueClassifier
from data import pathSpecDataset, audioSnippetDataset
import pickle, argparse, re, pdb, json, yaml, random, time, os, csv
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()
# metavar argument is for the actual name of the variable, in case the optional argument (eg. --batch-size) is not informative enough
parser.add_argument('--file_name', type=str, default='defaultName', metavar='N')
parser.add_argument('--model', type=str, default='luo', help='adjust code to work with Wilkins model and audio not spectrograms')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 4)')
parser.add_argument('--lstm_num', type=int, default=2, metavar='N', help='2 if not specified')
parser.add_argument('--use_attention', type=str2bool, default=True, help='')
parser.add_argument('--is_blstm', type=str2bool, default=True, help='')
parser.add_argument('--chunk_num', type=int, default=6, metavar='N', help='chunk_seconds is 23 if not specified')
parser.add_argument('--which_cuda', type=int, default=0, metavar='N')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N')
parser.add_argument('--seed', type=int, default=1, metavar='N')
parser.add_argument('--dropout', type=float, default=0., metavar='N')

parser.add_argument('--short', type=str2bool, default=False, help='adjust code to work with Wilkins model and audio not spectrograms')
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--chunk_seconds', type=float, default=0.5, metavar='N', help='chunk_seconds is 0.5 if not specified')
parser.add_argument('--ckpt_freq', type=int, default=100, metavar='N')
parser.add_argument('--load_ckpt', type=str, default='', metavar='N')
parser.add_argument('--reg', type=float, default=0, metavar='N')
parser.add_argument('--iteration', type=int, default=1, metavar='N')
parser.add_argument('--n_mels', type=int, default=96, metavar='N')
parser.add_argument('--data_dir', type=str, default='./spmel_desilenced_normalized', metavar='N')
parser.add_argument('--test_list', type=str, default='', metavar='N')
config = parser.parse_args()

config.cuda = not config.no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device(f"cuda:{config.which_cuda}" if config.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}
seconds = time.time()

results_dir = './results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
file_name_dir = './results/' +config.file_name
if not os.path.exists(file_name_dir):
    os.mkdir(file_name_dir)
results_csv='./results/' +config.file_name +'/RandomSearchReport.csv'
# record findings in csv
with open(results_csv, "w") as csvResults:
    csv_writer = csv.writer(csvResults)
    header = 'Title','lr','melNum','dropout','batchSize','reg','windowSize','bestEpoch','Loss','Acc'
    csv_writer.writerow(header)
    
    randomIterations=config.iteration
    for i in range(randomIterations):

        if randomIterations>1:
            # search parameters for random search
            config.lr = random.uniform(1e-6, 1e-2)
            config.n_mels = int(random.uniform(10, 79))
            config.dropout = random.uniform(0, 0.8)
            config.batch_size = random.randit(0, 4)
            config.reg = random.uniform(0, 0.01),
            config.chunk_seconds = int(random.uniform(23, 82))
        
        history_list=[[], [], [], []]

        string_config = json.dumps(vars(config))[1:-1]
        string_config = ''.join(e for e in string_config if e.isalnum())
        print(config)
        
        """For classification technique we can do a train/test split like in Wilkins, of 0.75"""
        # look at the amount of files in folder
        # randomly choose which singers will be used in training set
        # sort directory melspec by name and generate indices for training set and test set
        #if config.split_by == 'singer':
        print('here1', time.time() - seconds)
        m_list = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_']
        f_list = ['f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
        random.seed(config.seed)
        random.shuffle(m_list)
        random.shuffle(f_list)
        train_m_list, test_m_list = (m_list[:-3],m_list[-3:])
        train_f_list, test_f_list = (f_list[:-2],f_list[-2:])
        train_list = train_m_list + train_f_list
        test_list = test_m_list + test_f_list
        print('train_list', train_list)
        print('test_list', test_list)


        config.test_list = ' '.join(test_list)
        """ its too complex to write a universal specPathDataset with different subfolder structures.
        More ignostic to automatically upload from one shallow directory, and sort from there using the filename analysis.
        Make sure the dataset is fed data in same order as sorted fileList"""

        
        if config.model == 'wilkins':
            config.data_dir = './audio_path_list.pkl'
            dataset = audioSnippetDataset(config)
            fileList = pickle.load(open(config.data_dir, 'rb'))
        else:
            with open(config.data_dir +'/spmel_params.yaml') as File:
                spmel_params = yaml.load(File, Loader=yaml.FullLoader)
            _, _, fileList = next(os.walk(config.data_dir))
            dataset = pathSpecDataset(config, spmel_params)
        print('here2', time.time() - seconds)
        file_path_list = sorted(fileList)
        fileList = [os.path.basename(x) for x in file_path_list]
        train_indices_list = []
        test_indices_list = []

        for fileName_idx, fileName in enumerate(fileList):
            for substring in train_list:
                if substring in fileName:
                    train_indices_list.append(fileName_idx)
            for substring in test_list:
                if substring in fileName:
                    test_indices_list.append(fileName_idx)
        if config.short==True:
            train_indices_list = train_indices_list[:8]
            test_indices_list = test_indices_list[:4]
            config.batch_size = 2
            config.epochs=2
            config.num_chunks=2

#        now = datetime.now()
#        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        if config.file_name == 'defaultName' or config.file_name == 'deletable':
            writer = SummaryWriter('testRuns/test')
        else:
            writer = SummaryWriter(comment = '_' +config.file_name)

        tech_singer_labels = []
        pred_target_labels = []
        """https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets"""
        train_sampler = SubsetRandomSampler(train_indices_list) 
        test_sampler = SubsetRandomSampler(test_indices_list)   
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=False)
        test_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, drop_last=False)
        vt_classer = VoiceTechniqueClassifier(config)

        #example_data, example_targets, ex_singer_ids = iter(test_loader).next()
        #writer.add_graph(model, example_data.float())

        if config.load_ckpt != '':
            previous_epochs = int(re.findall('\d+', config.load_ckpt)[0])
        else: previous_epochs = 0
        for epoch in range(previous_epochs+1, previous_epochs+config.epochs+1):
            # history_list gets extended while inside these functions
            train_pred_target_labels, train_tech_singer_labels = vt_classer.infer(epoch, train_loader, history_list, writer, len(train_indices_list), 'train')
            val_pred_target_labels, val_tech_singer_labels = vt_classer.infer(epoch, test_loader, history_list, writer, len(test_indices_list), 'test')
            tech_singer_labels.append((train_tech_singer_labels, val_tech_singer_labels)) 
            pred_target_labels.append((train_pred_target_labels, val_pred_target_labels)) 
            writer.flush()

        writer.close() 
         # model_file_name = os.path.basename(__file__)[:-3]
        
        saveHistory(history_list, file_name_dir, string_config, tech_singer_labels, pred_target_labels)
        #invalid terms just
        bestLoss = history_list[0][0]

        for idx, loss in enumerate(history_list[0]):
            if loss <= bestLoss:
                bestLoss=loss
                bestEpoch=idx
        
        bestAcc = history_list[3][bestEpoch]
        
        with open(file_name_dir +'/config_params.pkl','wb') as File:
            pickle.dump(config, File) 
        
        csv_writer.writerow((string_config, config.lr, config.n_mels, config.dropout, config.batch_size, config.reg, config.chunk_seconds, bestEpoch, bestLoss, bestAcc))
csvResults.close(
)


