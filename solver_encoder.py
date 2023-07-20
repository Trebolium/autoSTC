from model_vc import Generator
from synthesis.model_bl import D_VECTOR
from vte_model import Vt_Embedder
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch
import math, os, pickle, sys
import utils
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time, pdb
import datetime

def save_ckpt(model, model_optimizer, loss, iteration, save_path):
    print('Saving model...')
    checkpoint = {'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': model_optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss}
    torch.save(checkpoint, save_path)

# writing this line as an excuse to update git message
class earlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.lowest_loss = 1000        

    def check(self, loss, model, model_optimizer, iteration, save_path):
        
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early stopping called at iteration {iteration}')
                save_ckpt(model, model_optimizer, loss, iteration, save_path)
                return True

# SOLVER IS THE MAIN SETUP FOR THE NN ARCHITECTURE. INSIDE SOLVER IS THE GENERATOR (G)
class Solver(object):

    def __init__(self, data_loader, config, spmel_params):
        """Initialize configurations.""" 
        self.config = config

        if self.config.file_name == 'defaultName' or self.config.file_name == 'deletable':
            self.writer = SummaryWriter('testRuns/test')
        else:
            self.writer = SummaryWriter(comment = '_' +self.config.file_name)

        self.spmel_params = spmel_params
        # Data loader.
        self.data_loader = data_loader

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.config.which_cuda}' if self.use_cuda else 'cpu')

        #self.singer_id_embs = torch.FloatTensor([embs[1] for embs in pickle.load(open('singer_id_embs.pkl','rb'))])
        # Build the model and tensorboard.
        self.build_model()
        self.earlyStopping = earlyStopping(self.config.patience)
        self.earlystop = False

        self.start_time = time.time()

    def build_model(self):

        if self.config.which_embs == 'vt-live' or self.config.which_embs == 'vt-avg':
            self.vte =  Vt_Embedder(self.config, self.spmel_params)
            for param in self.vte.parameters():
                param.requires_grad = False
            self.vte_optimizer = torch.optim.Adam(self.vte.parameters(), 0.0001)
            self.vte_checkpoint = torch.load(os.path.join(self.config.ste_path, 'best_epoch_checkpoint.pth.tar'))
            new_state_dict = OrderedDict()
            for i, (key, val) in enumerate(self.vte_checkpoint['model_state_dict'].items()):
#            if key.startswith('class_layer'):
#                continue
                new_state_dict[key] = val 
            self.vte.load_state_dict(new_state_dict)
            for state in self.vte_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.device)
            self.vte.to(self.device)
            self.vte.eval()
            self.avg_vt_embs = np.load(os.path.join(self.config.ste_path, 'averaged_embs.npy'))
            
        elif self.config.which_embs == 'spkrid-live':
            # C is the speaker encoder. The config values match with the paper
            self.C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
            # Speaker encoder checkpoint things. Load up the pretrained checkpoint info
            c_checkpoint = torch.load('/homes/bdoc3/my_data/autovc_data/3000000-BL.ckpt')
            new_state_dict = OrderedDict()
            for key, val in c_checkpoint['model_b'].items():
                new_key = key[7:]
                new_state_dict[new_key] = val 
            self.C.load_state_dict(new_state_dict)
               # freezes weights so they are unaffected by backprop
            for param in self.C.parameters():
                param.requires_grad = False
            self.C.to(self.device)
            
        self.G = Generator(self.config.dim_neck, self.config.dim_emb, self.config.dim_pre, self.config.freq)        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.adam_init)
        if self.config.ckpt_weights!='':
            ckpt_path = os.path.join(self.config.models_dir, self.config.ckpt_weights)
            g_checkpoint = torch.load(ckpt_path)
            self.G.load_state_dict(g_checkpoint['model_state_dict'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
            # fixes tensors on different devices error
            # https://github.com/pytorch/pytorch/issues/2830
            for state in self.g_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.current_iter = g_checkpoint['iteration']
            tester=2
        else:
            self.current_iter = 0
        self.G.to(self.device)


    def get_current_iters(self):
        return self.current_iter

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    def closeWriter(self):
        self.writer.close()
    
    #=====================================================================================================================================#
   
    def iterate(self, mode, data_loader, current_iter, cycle_size, log_list):

        def batch_iterate():
    
            for i in range(current_iter+1, (current_iter+1 + cycle_size)):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                try:
                    x_real, dataset_idx, example_id = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, dataset_idx, example_id = next(data_iter)
                #print(f'iter {i}, ds_idx {dataset_idx}, example_id {example_id}')
            
                x_real = x_real.to(self.device).float() 
                x_real_chunked = x_real.view(x_real.shape[0]*self.config.chunk_num, x_real.shape[1]//self.config.chunk_num, -1)
                # =================================================================================== #
                #                               2. Train the generator                                #
                # =================================================================================== #

                # DESIGNED ONLY FOR VCTK TESTS
                if self.config.which_embs == 'vt-live':
                    pred_style_idx, all_tensors = self.vte(x_real_chunked)
                    emb_org = all_tensors[-1]
#                elif self.config.which_embs == 'vt-avg':
#                    pred_output, all_tensors = self.vte(x_real_chunked)
#                    _, style_idx = torch.max(pred_output,1)
#                    emb_org = torch.tensor(self.avg_vt_embs[style_idx.cpu()]).to(self.device)
                elif self.config.which_embs == 'spkrid-live':
                    emb_org = self.C(x_real)
#                elif self.config.which_embs == 'spkrid-avg':
#                    emb_org = dataset_idx[1].to(self.device).float() # because Vctk datalaoder is configured this way 

                self.G = self.G.train()
                # x_identic_psnt consists of the original mel + the residual definiton added ontop
                x_identic, x_identic_psnt, code_real, _, _ = self.G(x_real, emb_org, emb_org)
                # SHAPES OF X_REAL AND X_INDETIC/PSNT ARE NOT THE SAME AND MAY GIVE INCORRECT LOSS VALUES
                residual_from_psnt = x_identic_psnt - x_identic
                x_identic = x_identic.squeeze(1)
                x_identic_psnt = x_identic_psnt.squeeze(1)
                residual_from_psnt = residual_from_psnt.squeeze(1)

                g_loss_id = F.l1_loss(x_real, x_identic)   
                g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)   
                
                # Code semantic loss. For calculating this, there is no target embedding
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                # gets the l1 loss between original encoder output and reconstructed encoder output
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                # Logging.
                loss = {}
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                loss['G/loss_cd'] = g_loss_cd.item()     
                losses_list[0] += g_loss_id.item()
                losses_list[1] += g_loss_id_psnt.item()
                losses_list[2] += g_loss_cd.item()

                # Print out training information.
                if i % self.config.log_step == 0 or i == (current_iter + cycle_size):
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    if mode == 'train':
                        log = "Elapsed [{}], Mode {}, Iter [{}/{}]".format(et, mode, i, self.config.max_iters)
                    else: log = "Elapsed [{}], Mode {}".format(et, mode)
                    for tag in keys:
                        log += ", {}: {:.4f}".format(tag, loss[tag])
                    print(log)
                    log_list.append(log)

                if mode == 'train':
                    # if self.config.with_cd ==True:
                    #     g_loss = (self.config.prnt_loss_weight * g_loss_id) + (self.config.psnt_loss_weight * g_loss_id_psnt) + (self.config.lambda_cd * g_loss_cd)
                    # else:
                    g_loss = (self.config.prnt_loss_weight * g_loss_id) + (self.config.psnt_loss_weight * g_loss_id_psnt) #+ ((self.config.lambda_cd  * (i / 100000)) * g_loss_cd)
                    
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    # spec nad freq have to be multiple of cycle_size
                    if i % self.config.spec_freq == 0:
                        print('plotting specs')
                        x_real = x_real.cpu().data.numpy()
                        x_identic = x_identic.cpu().data.numpy()
                        x_identic_psnt = x_identic_psnt.cpu().data.numpy()
                        residual_from_psnt = residual_from_psnt.cpu().data.numpy()
                        specs_list = []
                        for arr in x_real:
                            specs_list.append(arr)
                        for arr in x_identic:
                            specs_list.append(arr)
                        for arr in residual_from_psnt:
                            specs_list.append(arr)
                        for arr in x_identic_psnt:
                            specs_list.append(arr)
                        columns = 2
                        rows = 4
                        fig, axs = plt.subplots(4,2)
                        fig.tight_layout()
                        for j in range(0, columns*rows):
                            spec = np.rot90(specs_list[j])
                            fig.add_subplot(rows, columns, j+1)
                            if j == 5 or j == 6:
                                spec = spec - np.min(spec)
                                plt.clim(0,1)
                            plt.imshow(spec)
                            try:
                                name = 'Egs ' +str(example_id[j%2]) +', ds_idx ' +str(dataset_idx[j%2])
                            except:
                                pdb.set_trace()
                            plt.title(name)
                            plt.colorbar()
                        plt.savefig(self.config.models_dir +'/' +self.config.file_name +'/image_comparison/' +str(i) +'iterations')
                        plt.close()
        
                    ckpt_path = self.config.models_dir +'/' +self.config.file_name +'/ckpts/' +'ckpt_' +str(i) +'.pth.tar'
                    if i % self.config.ckpt_freq == 0:
                        save_ckpt(self.G, self.g_optimizer, loss, i, ckpt_path)
                
            return losses_list, (current_iter + cycle_size)

#=====================================================================================================================================#

        def logs(losses_list, mode, current_iter): 
            if mode[5:]=='vocal':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[5:]}/{mode[:5]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode[5:]=='medley':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[5:]}/{mode[:5]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode[5:]=='vctk':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[5:]}/{mode[:5]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode=='train':
                self.writer.add_scalar(f"Loss_id_psnt_{self.config.use_loader}/{mode}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            else: exit(1)
            losses_list = [0.,0.,0.]
            self.writer.flush()
            print('writer flushed')
#            if mode == 'test':
#                es_ckpt_path = self.config.models_dir +'/' +self.config.file_name +'/ckpts/' +'ckpt_' +str(current_iter) +'_earlyStop.pth.tar'
#                self.earlystop = self.earlyStopping.check(losses_list[0], self.G, self.g_optimizer, current_iter, es_ckpt_path)
#                if self.earlystop == True:
#                    sys.exit(1)
#=====================================================================================================================================#

        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        last_save = file_path = hist_file_path =  'delete.txt' 
        losses_list = [0., 0., 0.]
        # Start training.
        print('Start training...')

        if mode == 'train':
            self.G.train()
#            loss_hist=history_list[0]
#            acc_hist=history_list[1]
            losses_list, current_iter = batch_iterate()
            logs(losses_list, mode, current_iter)
        elif mode.startswith('test'):
            best_acc = 0 
            self.G.eval()
#            loss_hist=history_list[2]
#            acc_hist=history_list[3]
            with torch.no_grad():
                losses_list, _ = batch_iterate()
            logs(losses_list, mode, current_iter)

        return current_iter, log_list            
