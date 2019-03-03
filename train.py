import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pickle
import argparse


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        '--lr',type=float, required=True)
    parser.add_argument(
        '--momentum',type=float,required=True)
    parser.add_argument(
        '--num_hidden',type=int,required=True)
    parser.add_argument(
        '--sizes',type=str,required=True)
    parser.add_argument(
        '--activation',type=str,required=True)
    parser.add_argument(
        '--loss',type=str,required=True)
    parser.add_argument(
        '--opt',type=str,required=True)
    parser.add_argument(
        '--batch_size',type=int,required=True)
    parser.add_argument(
        '--epochs',type=int,required=True)
    parser.add_argument(
        '--anneal',type=str,required=True)
    parser.add_argument(
        '--save_dir',type=str,required=True)
    parser.add_argument(
        '--expt_dir',type=str,required=True)
    parser.add_argument(
        '--train',type=str,required=True)
    parser.add_argument(
        '--val',type=str,required=True)
    parser.add_argument(
        '--test',type=str,required=True)
    parser.add_argument(
        '--pretrain',type=str,required=True)
    parser.add_argument(
        '--state',type=int,required=True)
    parser.add_argument(
        '--testing',type=str,required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    lr = args.lr
    momentum = args.momentum
    num_hidden=args.num_hidden
    sizes = [int(item) for item in args.sizes.split(',')]
    activation=args.activation
    loss=args.loss
    opt=args.opt
    batch_size=args.batch_size
    epochs=args.epochs
    anneal=args.anneal
    save_dir=args.save_dir
    expt_dir=args.expt_dir
    train=args.train
    val=args.val
    test=args.test
    pretrain=args.pretrain
    state=args.state
    testing=args.testing
    
    # Return all variable values
    return lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epochs,anneal,save_dir,expt_dir,train,val,test,pretrain,state,testing

# Run get_args()
# get_args()

# Match return values from get_arguments()
# and assign to their respective variables
# server, port, keyword = get_args()
lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epochs,anneal,save_dir,expt_dir,train,val,test,pretrain,state,testing = get_args()
ann=False
if(anneal=="true" or anneal=="True"):
    ann=True
prtr=False
if(pretrain=="true" or pretrain=="True"):
    prtr=True
tstn=False
if(testing=="true" or testing=="True"):
    tstn=True
pretrain=prtr
anneal=ann
testing=tstn

from sklearn.decomposition import PCA

def log_file(expt_dir,tv,infor):
    f=open(expt_dir + tv,'a')
    f.write(infor+"\n")
    f.close()

def save_weights(list_of_weights, epoch,save_dir):
    with open(save_dir+'weights_{}.pkl'.format(epoch), 'wb') as f:
        pickle.dump(list_of_weights, f)

def load_weights(state,save_dir):
    with open(save_dir+'weights_{}.pkl'.format(state),'rb') as f:
        list_of_weights = pickle.load(f)
    return list_of_weights

def soft_max(y):
    az=np.copy(y)
    az=az-np.max(az)
    a=np.exp(az)
    summ = np.reshape(np.sum(a,axis=1),(a.shape[0],1))
    summ[summ==0]=1
    a=a/summ
    return a

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    powe = np.exp(-2*z)
    return (1-powe )/(1+powe)

def relu(z):
    z[z<=0]=0.01*z[z<=0]
    return (z)

def grad_sig(h):
    return h*(1-h)

def grad_tanh(h):
    return (1-h*h)

def grad_relu(h):
    ah=np.copy(h)
    ah[ah>0]=1
    ah[ah<=0]=0.01
    return (ah)

def acti(activation,z):
    if activation=='sigmoid':
        return sigmoid(z)
    elif activation=='tanh':
        return tanh(z)
    else:
        return relu(z)

def gradi(activation,h):
    if activation=='sigmoid':
        return grad_sig(h)
    elif activation=='tanh':
        return grad_tanh(h)
    else:
        return grad_relu(h)


def gen_plot(lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epochs,anneal,save_dir,expt_dir,train,val,test,pretrain,state,testing):
    feat = 100
#     lr = 0.01
#     momentum = 0.9
#    num_hidden = 1
#    sizes = [feat,10,10]
#    activation = 'sigmoid'
#    loss = 'ce'
#    opt = 'nag'
#    batch_size = 30
#    anneal = True
#     epochs=30
#     pretrain = False
#     state = 16
#     testing = False
    if testing==True:
        epochs=0
#     save_dir = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/Programming Assignment 1/save_dir/"
#     expt_dir = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/Programming Assignment 1/expt_dir/"
#     train = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/Programming Assignment 1/train.csv"
#     test = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/Programming Assignment 1/test.csv"
#     val = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/Programming Assignment 1/valid.csv"    
    
    appp=[feat]
    for i in range(num_hidden):
        appp.append(sizes[i])
    appp.append(10)
    sizes = np.copy(appp)
    train = pd.read_csv(train)
    valid = pd.read_csv(val)
    test = pd.read_csv(test)
    train = train.drop('id',axis=1)
    valid=valid.drop('id',axis=1)
    test = test.drop('id',axis=1)
    
    np.random.seed(1234)
    x_train = np.asarray(train.loc[:, train.columns != 'label'])
    divi = np.max(x_train,axis=0)-np.min(x_train,axis=0)
    means = np.mean(x_train,axis=0)
    x_train=x_train-means
#     vari = np.sum(x_train**2,axis=0)/x_train.shape[0]
#     vari = np.sqrt(vari)
    x_train = x_train/divi
    y_train = np.asarray(train['label'])
    
    x_valid = np.asarray(valid.loc[:, valid.columns != 'label'])
    x_valid = x_valid-means
    x_valid = x_valid/divi
    y_valid = np.asarray(valid['label'])
    
    x_test = np.asarray(test.loc[:, valid.columns != 'label'])
    x_test = x_test-means
    x_test = x_test/divi
    
    pca = PCA(n_components=feat)
    x_train = pca.fit_transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)
    #x_train = x_train[0:55000]
    #y_train = y_train[0:55000]
    
    
    m=x_train.shape[0]
    siz = 0
    siz1=0
    pos=[]
    pos1=[]
    pos.append(0)
    pos1.append(0)
    for i in range(len(sizes)-1):
        siz=siz+sizes[i]*sizes[i+1]
        siz1=siz1+sizes[i+1]
        pos.append(siz)
        pos1.append(siz1)
    if pretrain==True:
        [wt,bt,gamma,beta]=load_weights(state,save_dir)
        e=state
    else:
        wt=np.random.randn(siz)*np.sqrt(2.0/m)
        bt=np.random.randn(siz1)*np.sqrt(2.0/m)
        gamma=np.random.randn(siz1)*np.sqrt(2.0/m)
        beta=np.random.randn(siz1)*np.sqrt(2.0/m)
        e=0
    lamb=0.01
    batch_norm=False
    tr_lo1,va_lo1=[],[]
    
    #momentum,nag_based
    prev_vw,prev_vb = 0,0
    #adam
    mw,mb,vw,vb,mw_hat,mb_hat,vw_hat,vb_hat,eps,beta1,beta2=0,0,0,0,0,0,0,0,1e-8,0.9,0.999
    x1 = np.copy(x_train)
    y1 = np.copy(y_train)
    accuracy_v=0
    accuracy=0
    
#    dots=np.ceil(x1.shape[0]/batch_size)
#    valid_prev_loss=100

    accuracy_v_prev=0
    wt_prev=np.copy(wt)
    bt_prev=np.copy(bt)
    mw_prev,mb_prev,vw_prev,vb_prev,mw_hat_prev,mb_hat_prev,vw_hat_prev,vb_hat_prev,prev_vw_prev,prev_vb_prev=0,0,0,0,0,0,0,0,0,0
#    flag=0
#    flag1=0
    flag2=0
#     e=epochs-1
    step=0
    while(e<epochs):
    #    if(accuracy>90 and flag1==0):
    #        lamb=0.01
    #        lr=0.1
    #        flag1=1
    #    if(accuracy_v>84 and flag==0):
    #        batch_size*=10
    #        flag=1
        m=batch_size
        for bat_nm in range(int(np.ceil(x1.shape[0]/batch_size))):
            step+=1
            x_train = x1[bat_nm*batch_size:min((bat_nm+1)*batch_size,x1.shape[0])]
            y_train = y1[bat_nm*batch_size:min((bat_nm+1)*batch_size,x1.shape[0])]
            h=np.copy(x_train)
            act,act1,act2,act3,act4=[],[],[],[],[]
            act.append(x_train)
            a=[]
            for i in range(len(sizes)-1):
                w=np.reshape(wt[pos[i]:pos[i+1]],(sizes[i+1],sizes[i]))
                a=np.matmul(h,np.transpose(w))
                a=a+np.reshape(bt[pos1[i]:pos1[i+1]],(sizes[i+1],))
                #test.append(a)
                #h=sigmoid(a)
                h=acti(activation,a)
                if batch_norm==True:
                    ylin,h_norm,mea,var=normalize(h,np.reshape(gamma[pos1[i]:pos1[i+1]],(sizes[i+1],)),np.reshape(beta[pos1[i]:pos1[i+1]],(sizes[i+1],)))
                    act1.append(ylin)
                    act2.append(h_norm)
                    act3.append(mea)
                    act4.append(var)
                act.append(h)
                if batch_norm==True:
                    h=np.copy(ylin)
            y_hat = soft_max(a)
            y_pred = np.argmax(y_hat,axis=1)
            train_loss=0
            #print(y_hat[0])
#             for i in range(x_train.shape[0]):
#                 train_loss+=(-1)*(np.matmul(np.log(y_hat[i]),np.transpose(np.eye(10)[y_train[i]])))/(x_train.shape[0])
            #print("epoch train_loss batch",e,bat_nm,train_loss)
            accuracy = 100*np.sum(y_pred==y_train)/x_train.shape[0]
            #print("epoch accuracy batch",e,bat_nm,accuracy)
            l=len(pos)
            del act[-1]
            
            sum_gw=np.zeros(siz)
            sum_gb=np.zeros(siz1)
            sum_gmea=np.zeros(siz1)
            sum_gvar=np.zeros(siz1)
            sum_gamma=np.zeros(siz1)
            sum_gbeta=np.zeros(siz1)
            
            if opt=='nag':
                wt=wt-momentum*prev_vw
                bt=bt-momentum*prev_vb
            for i in range(x_train.shape[0]):
                grad_a = -(np.eye(10)[y_train[i]]-y_hat[i])
                if loss=='sq':
                    grad_a = np.matmul((np.identity(10)*y_hat[i]-np.reshape(y_hat[i],(10,1))*np.reshape(y_hat[i],(1,10))),(y_hat[i]-np.eye(10)[y_train[i]]))
                for j in range(len(sizes)-1):
                    w=np.reshape(wt[pos[l-j-2]:pos[l-j-1]],(sizes[l-j-1],sizes[l-j-2]))
                    grad_w=np.matmul(np.reshape(grad_a,(len(grad_a),1)),np.transpose(np.reshape(act[l-j-2][i],(len(act[l-j-2][i]),1))))
                    grad_b=np.copy(grad_a)
                    grad_h=np.matmul(np.transpose(w),grad_a)
                    if batch_norm==True:
                        grad_h,dgamma,dbeta=grad_batchnorm(grad_h,h,h_norm,mea,var,gamma,beta)
                    grad_a=grad_h*gradi(activation,act[l-j-2][i])
                    sum_gw[pos[l-j-2]:pos[l-j-1]]=sum_gw[pos[l-j-2]:pos[l-j-1]]+np.reshape(grad_w,(sizes[l-j-2]*sizes[l-j-1],))
                    sum_gb[pos1[l-j-2]:pos1[l-j-1]]=sum_gb[pos1[l-j-2]:pos1[l-j-1]]+grad_b
            sum_gw+=2*lamb*wt
            sum_gb+=2*lamb*bt
            if opt=='gd':
                wt=wt-lr*sum_gw/m
                bt=bt-lr*sum_gb/m
            elif opt=='momentum' or opt=='nag':
                vw=momentum*prev_vw+lr*sum_gw/m
                vb=momentum*prev_vb+lr*sum_gb/m
                wt=wt-vw
                bt=bt-vb
                prev_vw=np.copy(vw)
                prev_vb=np.copy(vb)
            elif opt=='adam':
                mw=beta1*mw+(1-beta1)*sum_gw/m
                mb=beta1*mb+(1-beta1)*sum_gb/m
                vw=beta2*vw+(1-beta2)*(sum_gw/m)**2
                vb=beta2*vb+(1-beta2)*(sum_gb/m)**2
                mw_hat=mw/(1-math.pow(beta1,e+1))
                mb_hat=mb/(1-math.pow(beta1,e+1))
                vw_hat=vw/(1-math.pow(beta2,e+1))
                vb_hat=vb/(1-math.pow(beta2,e+1))
                wt=wt-(lr/np.sqrt(vw_hat+eps))*mw_hat
                bt=bt-(lr/np.sqrt(vb_hat+eps))*mb_hat
            if step%100==0:
                h_v=np.copy(x1)
                a_v=[]
                for i in range(len(sizes)-1):
                    w=np.reshape(wt[pos[i]:pos[i+1]],(sizes[i+1],sizes[i]))
                    a_v=np.matmul(h_v,np.transpose(w))
                    a_v=a_v+np.reshape(bt[pos1[i]:pos1[i+1]],(sizes[i+1],))
                    h_v=acti(activation,a_v)
                y_hat_v = soft_max(a_v)
                train_loss=0
                if loss=='ce':
                    for i in range(x1.shape[0]):
                        train_loss+=(-1)*(np.matmul(np.log(y_hat_v[i]),np.transpose(np.eye(10)[y1[i]])))/(x1.shape[0])
                else:
                    for i in range(x1.shape[0]):
                        train_loss+=np.sum((y_hat_v[i]-np.eye(10)[y1[i]])**2)/((x1.shape[0])*10.0)

#                 print("Training Loss ",e,train_loss)
                y_pred = np.argmax(y_hat_v,axis=1)
                accuracy = 100*np.sum(y_pred==y1)/x1.shape[0]
#                 print("Training accuracy",e,accuracy)
                train_file="log_train.txt"
                infor="Epoch "+str(e)+",\tStep "+str(step)+",\tLoss: "+str(train_loss)+",\tError: "+str((100-accuracy))+",\tlr: "+str(lr)
                log_file(expt_dir,train_file,infor)
                
                h_v=np.copy(x_valid)
                act_v=[]
                act_v.append(x_valid)
                a_v=[]
                for i in range(len(sizes)-1):
                    w=np.reshape(wt[pos[i]:pos[i+1]],(sizes[i+1],sizes[i]))
                    a_v=np.matmul(h_v,np.transpose(w))
                    a_v=a_v+np.reshape(bt[pos1[i]:pos1[i+1]],(sizes[i+1],))
                    h_v=acti(activation,a_v)
                    act_v.append(h_v)
                y_hat_v = soft_max(a_v)
                valid_loss=0
                if loss=='ce':
                    for i in range(x_valid.shape[0]):
                        valid_loss+=(-1)*(np.matmul(np.log(y_hat_v[i]),np.transpose(np.eye(10)[y_valid[i]])))/(x_valid.shape[0])
                else:
                    for i in range(x_valid.shape[0]):
                        valid_loss+=np.sum((y_hat_v[i]-np.eye(10)[y_valid[i]])**2)/((x_valid.shape[0])*10.0)

                y_pred_v = np.argmax(y_hat_v,axis=1)
#                 print("epoch valid loss",e,valid_loss)
                accuracy_v= 100*np.sum(y_pred_v==y_valid)/x_valid.shape[0]
#                 print("epoch valid accuracy",e,accuracy_v)
                infor="Epoch "+str(e)+",\tStep "+str(step)+",\tLoss: "+str(train_loss)+",\tError: "+str((100-accuracy))+",\tlr: "+str(lr)
                val_file="log_val.txt"
                log_file(expt_dir,val_file,infor)
        h_v=np.copy(x1)
        a_v=[]
        for i in range(len(sizes)-1):
            w=np.reshape(wt[pos[i]:pos[i+1]],(sizes[i+1],sizes[i]))
            a_v=np.matmul(h_v,np.transpose(w))
            a_v=a_v+np.reshape(bt[pos1[i]:pos1[i+1]],(sizes[i+1],))
            h_v=acti(activation,a_v)
        y_hat_v = soft_max(a_v)
        train_loss=0
        if loss=='ce':
            for i in range(x1.shape[0]):
                train_loss+=(-1)*(np.matmul(np.log(y_hat_v[i]),np.transpose(np.eye(10)[y1[i]])))/(x1.shape[0])
        else:
            for i in range(x1.shape[0]):
                train_loss+=np.sum((y_hat_v[i]-np.eye(10)[y1[i]])**2)/((x1.shape[0])*10.0)
        
#         print("Training Loss ",e,train_loss)
        y_pred = np.argmax(y_hat_v,axis=1)
        accuracy = 100*np.sum(y_pred==y1)/x1.shape[0]
#         print("Training accuracy",e,accuracy)
        
        h_v=np.copy(x_valid)
        act_v=[]
        act_v.append(x_valid)
        a_v=[]
        for i in range(len(sizes)-1):
            w=np.reshape(wt[pos[i]:pos[i+1]],(sizes[i+1],sizes[i]))
            a_v=np.matmul(h_v,np.transpose(w))
            a_v=a_v+np.reshape(bt[pos1[i]:pos1[i+1]],(sizes[i+1],))
            h_v=acti(activation,a_v)
            act_v.append(h_v)
        y_hat_v = soft_max(a_v)
        valid_loss=0
        if loss=='ce':
            for i in range(x_valid.shape[0]):
                valid_loss+=(-1)*(np.matmul(np.log(y_hat_v[i]),np.transpose(np.eye(10)[y_valid[i]])))/(x_valid.shape[0])
        else:
            for i in range(x_valid.shape[0]):
                valid_loss+=np.sum((y_hat_v[i]-np.eye(10)[y_valid[i]])**2)/((x_valid.shape[0])*10.0)
    
        y_pred_v = np.argmax(y_hat_v,axis=1)
#         print("epoch valid loss",e,valid_loss)
        accuracy_v= 100*np.sum(y_pred_v==y_valid)/x_valid.shape[0]
#         print("epoch valid accuracy",e,accuracy_v)
        if(anneal==True and accuracy_v < accuracy_v_prev):
#             if flag2==5:
#                 e=epochs
            lr/=2.0
            wt=np.copy(wt_prev)
            bt=np.copy(bt_prev)
            mw=np.copy(mw_prev)
            mb=np.copy(mb_prev)
            vw=np.copy(vw_prev)
            vb=np.copy(vb_prev)
            prev_vw=np.copy(prev_vw_prev)
            prev_vb=np.copy(prev_vb_prev)
            mw_hat=np.copy(mw_hat_prev)
            mb_hat=np.copy(mb_hat_prev)
            vw_hat=np.copy(vw_hat_prev)
            vb_hat=np.copy(vb_hat_prev)
            print("wasted")
            flag2+=1
            e+=1
            continue
        else:
            tr_lo1.append(train_loss)
            va_lo1.append(valid_loss)
        wt_prev=np.copy(wt)
        bt_prev=np.copy(bt)
        accuracy_v_prev=accuracy_v
        mw_prev=np.copy(mw)
        mb_prev=np.copy(mb)
        vw_prev=np.copy(vw)
        vb_prev=np.copy(vb)
        prev_vw_prev=np.copy(prev_vw)
        prev_vb_prev=np.copy(prev_vb)
        mw_hat_prev=np.copy(mw_hat)
        mb_hat_prev=np.copy(mb_hat)
        vw_hat_prev=np.copy(vw_hat)
        vb_hat_prev=np.copy(vb_hat)
        e+=1
        save_weights([wt,bt],e,save_dir)
#        lr = lr*e/(e+1)
            #progress[bat_nm]="="
            #print(progress)
    h_test=np.copy(x_test)
    a_test=[]
    for i in range(len(sizes)-1):
        w=np.reshape(wt[pos[i]:pos[i+1]],(sizes[i+1],sizes[i]))
        a_test=np.matmul(h_test,np.transpose(w))
        a_test=a_test+np.reshape(bt[pos1[i]:pos1[i+1]],(sizes[i+1],))
        h_test=acti(activation,a_test)
    y_hat1 = soft_max(a_test)
    y_pred = np.argmax(y_hat1,axis=1)
    
    y_pred_test = np.zeros([y_pred.shape[0],2],dtype=int)
    for i in range (y_pred.shape[0]):
        y_pred_test[i][0]=i
        y_pred_test[i][1]=y_pred[i]
    
#     df = pd.DataFrame(y_pred_test,columns=['id','label'])
#     df.to_csv(r"C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/Programming Assignment 1/submission.csv",index=False)

    return tr_lo1,va_lo1,df

gen_plot(lr,momentum,num_hidden,sizes,activation,loss,opt,batch_size,epochs,anneal,save_dir,expt_dir,train,val,test,pretrain,state,testing)