import argparse,os,re
import configparser
import sys

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--epochs', type=int, default=21, help='')

    parser.add_argument('--ascc_mode', type=str, default='comb_p', help='')

    parser.add_argument('--work_path', type=str, default='./', help='')

    parser.add_argument('--smooth_ce', type=str, default='false', help='')

    parser.add_argument('--if_ce_adp', type=str, default='false', help='')
    
    parser.add_argument('--h_test_start', type=int, default=0, help='')

    parser.add_argument('--vis_w_key_token', type=int, default=None, help='')

    parser.add_argument('--snli_epochs', type=int, default=81, help='')

    parser.add_argument('--resume_vector_only', type=str, default='false', help='')

    parser.add_argument('--l2_ball_range', type=str, default='sentence', help='')

    parser.add_argument('--normalize_embedding', type=str, default='false', help='')

    parser.add_argument('--lm_constraint', type=str, default='true', help='')

    parser.add_argument('--label_smooth', type=float, default=0, help='')

    parser.add_argument('--imdb_lm_file_path', type=str, default="lm_scores/imdb_all.txt", help='') 
    parser.add_argument('--snli_lm_file_path', type=str, default="lm_scores/snli_all_save.txt", help='') 

    parser.add_argument('--embd_freeze', type=str, default='false', help='')

    parser.add_argument('--embd_fc_freeze', type=str, default='false', help='')

    parser.add_argument('--embd_transform', type=str, default='true', help='')

    parser.add_argument('--certified_neighbors_file_path', type=str, default="counterfitted_neighbors.json", help='') 

    parser.add_argument('--train_attack_sparse_weight', type=float, default=15, help='') 

    parser.add_argument('--attack_sparse_weight', type=float, default=15, help='') 

    parser.add_argument('--w_optm_lr', type=float, default=10, help='') 

    parser.add_argument('--bert_w_optm_lr', type=float, default=1, help='') 
    
    parser.add_argument('--pert_set', type=str, default='ad_text_syn_p', help='')

    parser.add_argument('--ge_file_path', type=str, default='../../GraphEmbedding/examples/walk_embeddings_it2.pickle', help='')

    parser.add_argument('--out_syn_netx_file', type=str, default='false', help='')

    parser.add_argument('--freeze_bert_stu', type=str, default='false', help='') 
    parser.add_argument('--freeze_bert_tea', type=str, default='true', help='') 

    parser.add_argument('--atten_kl_weight', type=float, default=0, help='') 
    
    parser.add_argument('--resume', type=str, default=None, help='') 

    parser.add_argument('--pwws_test_num', type=int, default=1000, help='') 
    parser.add_argument('--genetic_test_num', type=int, default=1000, help='') 
    parser.add_argument('--genetic_iters', type=int, default=40, help='') 
    parser.add_argument('--genetic_pop_size', type=int, default=60, help='') 

    parser.add_argument('--kl_start_epoch', type=int, default=0, help='') 

    parser.add_argument('--weight_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_ball', type=float, default=0, help='') 
    parser.add_argument('--weight_kl', type=float, default=0, help='') 
    
    parser.add_argument('--weight_mi_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_giveny_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_giveny_adv', type=float, default=0, help='') 
    
    parser.add_argument('--infonce_sim_metric', type=str, default='0429', help='') 
       
    parser.add_argument('--bert_logit_use_RP_tea', type=str, default='false', help='') 

    parser.add_argument('--new_exp', type=str, default='false', help='')

    parser.add_argument('--info_input_dim', type=int, default=16, help='') 

    parser.add_argument('--test_attack_iters', type=int, default=10,
                    help='') 
    parser.add_argument('--test_attack_eps', type=float, default=1,
                    help='') 
    parser.add_argument('--test_attack_step_size', type=float, default=0.25,
                    help='') 

    parser.add_argument('--random_start', type=str, default='true', help='')

    parser.add_argument('--train_attack_iters', type=int, default=10,
                    help='') 
                    
    parser.add_argument('--train_attack_eps', type=float, default=5.0, help='') 

    parser.add_argument('--train_attack_step_size', type=float, default=0.25,
                    help='') 

    parser.add_argument('--use_pretrained_embeddings', type=str, default='true', help='')

    parser.add_argument('--imdb_synonyms_file_path', type=str, default="temp/imdb.synonyms",
                    help='')

    parser.add_argument('--imdb_bert_synonyms_file_path', type=str, default="temp/imdb.bert.synonyms",
                    help='')


    parser.add_argument('--snli_synonyms_file_path', type=str, default="temp/snli.synonyms",
                    help='')

    parser.add_argument('--snli_bert_synonyms_file_path', type=str, default="temp/snli.bert.synonyms",
                    help='')

    parser.add_argument('--synonyms_from_file', type=str, default='true',
                    help='')
                    
    parser.add_argument('--bow_mean', type=str, default='false', help='')

    parser.add_argument('--embedding_prep', type=str, default='ori', help='')

    parser.add_argument('--out_path', type=str, default="./",
                    help='')
                    
    parser.add_argument('--train_mode', type=str, default="set_radius_ad",
                    help='')

    parser.add_argument('--config', type=str, default="no_file_exists",
                    help='gpu number')
        
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')     

    parser.add_argument('--max_seq_len', type=int, default=400,
                    help='max_seq_len')
                    
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size') 

    parser.add_argument('--test_batch_size', type=int, default=32,
                    help='test_batch_size') 

    parser.add_argument('--syn_batch_size', type=int, default=2048,
                    help='syn_batch_size')

    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding_dim')

    parser.add_argument('--embedding_out_dim', type=int, default=100, help='embedding_dim')
    
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning_rate')

    parser.add_argument('--ball_learning_rate', type=float, default=0.01, help='learning_rate')

    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight_decay')
    parser.add_argument('--ball_weight_decay', type=float, default=2e-4, help='weight_decay')

    parser.add_argument('--optimizer', type=str, default="adamw", help='optimizer')

    parser.add_argument('--ball_optimizer', type=str, default="sgd", help='ball_optimizer')

    parser.add_argument('--lr_scheduler', type=str, default="none", help='lr_scheduler')

    parser.add_argument('--grad_clip', type=float, default=1e-1, help='grad_clip')
            
    parser.add_argument('--RP_learnable', type=str, default="RPstu_and_RPtea", help='')

    parser.add_argument('--invertible_RP_tea', type=str, default="false", help='')


    parser.add_argument('--model', type=str, default="cnn_adv",
                    help='model name')

    parser.add_argument('--dataset', type=str, default="imdb",
                    help='dataset')

    parser.add_argument('--position', type=bool, default=False,
                    help='gpu number')
    
    parser.add_argument('--keep_dropout', type=float, default=0.2,
                    help='keep_dropout')

    parser.add_argument('--embedding_file_path', type=str, default="glove/glove.840B.300d.txt",
                    help='glove or w2v')

    parser.add_argument('--embedding_training', type=str, default="false",
                    help='embedding_training')
    #kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                    help='kernel_sizes')

    parser.add_argument('--info_layers', type=str, default="12,",
                    help='')

    parser.add_argument('--lle_k', type=int, default=8,
                    help='')

    parser.add_argument('--dist_kl_tau', type=float, default=1,
                    help='')

    parser.add_argument('--dist_p_type', type=str, default="pjci",
                    help='')

    parser.add_argument('--dist_div_type', type=str, default="kl",
                    help='')

    parser.add_argument('--h_type', type=str, default="discriminator",
                    help='')

    parser.add_argument('--save_bert_fea', type=str, default="false",
                    help='')
            
    parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                    help='kernel_nums')
    parser.add_argument('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    parser.add_argument('--lstm_mean', type=str, default="mean",# last
                    help='lstm_mean')
    parser.add_argument('--lstm_layers', type=int, default=1,# last
                    help='lstm_layers')
    parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')
    parser.add_argument('--gpu_num', type=int, default=1,
                    help='gpu number')
    parser.add_argument('--proxy', type=str, default="null",
                    help='http://proxy.xx.com:8080')

    parser.add_argument('--debug', type=bool, default=False,
                    help='')

    parser.add_argument('--bidirectional', type=str, default="true",
                    help='bidirectional')
    
    parser.add_argument('--embedding_dir', type=str, default=".glove/glove.6B.300d.txt",
                    help='embedding_dir')
    
#
    args = parser.parse_args()
    
    if args.config != "no_file_exists":
        if os.path.exists(args.config):
            config = configparser.ConfigParser()
            config_file_path=args.config
            config.read(config_file_path)
            config_common = config['COMMON']
            for key in config_common.keys():
                args.__dict__[key]=config_common[key]
        else:
            print("config file named %s does not exist" % args.config)

#        args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
#        args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
    
    # process the type for bool and list    
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg])==str:
            if args.__dict__[arg].lower()=="true":
                args.__dict__[arg]=True
            elif args.__dict__[arg].lower()=="false":
                args.__dict__[arg]=False
            elif "," in args.__dict__[arg]:
                args.__dict__[arg]= [int(i) for i in args.__dict__[arg].split(",") if i!='']
            else:
                pass

    sys.path.append(args.work_path)

    for arg in args.__dict__.keys():
        if "path" in arg and arg!="work_path":
            args.__dict__[arg] = os.path.join(args.work_path, args.__dict__[arg])

    if os.path.exists("proxy.config"):
        with open("proxy.config") as f:
            args.proxy = f.read()
            print(args.proxy)
    
    return args 