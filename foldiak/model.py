import torch
from torch.utils.data import DataLoader


class Foldiak:
    def __init__(
            self,
            n_features:int,
            n_neurons:int=16,
            ylr:float=5e-2,
            n_iterations:int=300,
            p:float=0.25,
            lmbda:float=10,
            device=torch.device('cpu')
            ):
        """Foldiak model initializing function

        Parameters
        ----------
        n_features : int
            number of features in data
        n_neurons : int
            number of neurons to model
        ylr : float
            step size of neural activities, y
        n_interations : int
            number of steps in simulating neural activities
        p : float
            probably of neuron being active 
        lmbda : float
            scale of sigmoid function
        device : torch.device
            device to do learning on
        """
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.ylr = ylr
        self.n_iterations = n_iterations
        self.p = p
        self.lmbda = lmbda
        self.device = device

        self.thresh = torch.zeros([n_neurons,]).to(device)
        self.q = torch.empty([n_features,n_neurons]).normal_().to(device)
        self.w = torch.zeros([n_neurons,n_neurons]).to(device)
        self.normalize_q()


    def f(self,x):
        """Sigmoid nonlinearity

        Parameters
        ----------
        x : tensor

        Returns 
        -------
        tensor of same shape through sigmoid
        """
        return torch.sigmoid(self.lmbda*x)
        # return 1/(1+torch.exp(-10*x))
    

    def inference(self,batch):
        """Compute neural activities

        Parameters
        ----------
        batch : shape=[batch_size,n_features]
            input data

        Returns 
        -------
        neural activities : shape=[batch_size,n_neurons]
        """
        n_samples = batch.shape[0]
        ystar = torch.zeros([n_samples,self.n_neurons]).to(self.device)
        for j in range(self.n_iterations):
            # grad update on y*,
            delta_ystar = self.f(batch@self.q + ystar@self.w - self.thresh) - ystar
            ystar += self.ylr*delta_ystar
        # threshold
        y = torch.where(ystar>0.5,1,0)
        return y


    def learn_model_weights(self,data,n_epoch,warm_up_epochs,batch_size,w_lr=0.1,q_lr=0.02,t_lr=0.02):
        """Learning scheme for model

        Parameters
        ----------
        data : shape=[dataset,n_features]
            input data
        n_epoch : int
            number of loops through data
        warm_up_epochs : int
            number of epochs to only update threshold
        batch_size : int 

        w_lr : float
            step size for inhibitory feedback weights, w
        q_lr : float 
            step size for feedforward weights, q 
        t_lr : float
            step size for thresholds, t
        """
        for i in (range(n_epoch)):

            alpha_,beta_,gamma_ = 0, 0, 0.1
            if i > warm_up_epochs:
                alpha_,beta_,gamma_ = w_lr,q_lr,t_lr
            
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
            iterloader = iter(dataloader)
            
            for batch in iterloader:
            
                y = self.inference(batch)

                # grad update on w
                deltaw = -alpha_*torch.mean((torch.einsum('bi,bj->bij',y,y) - self.p**2),dim=0)
                deltaw.fill_diagonal_(0)
                deltaw[deltaw>0] = 0
                self.w += deltaw

                # grad update on q
                deltaq = beta_*torch.mean((batch[:,:,None]*y[:,None,:] - y.reshape([batch.shape[0],1,self.n_neurons])*self.q),dim=0)
                self.q += deltaq

                self.normalize_q()
                
                # grad update on thresh
                deltathresh = gamma_*torch.mean((y-self.p),dim=0)
                self.thresh += deltathresh                


    def normalize_q(self):
        """
        normalize receptive fields of feedforward weights
        """
        self.q = self.q/self.q.norm(dim=0,keepdim=True)