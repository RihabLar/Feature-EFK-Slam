from MapFeature import *
from FEKFMBL import *
from scipy.linalg import block_diag

class FEKFSLAM(FEKFMBL):
    """
    Class implementing the Feature-based Extended Kalman Filter for Simultaneous Localization and Mapping (FEKFSLAM).
    It inherits from the FEKFMBL class, which implements the Feature Map based Localization (MBL) algorithm using an EKF.
    :class:`FEKFSLAM` extends :class:`FEKFMBL` by adding the capability to map features previously unknown to the robot.
    """
 
    def __init__(self,  *args):

        super().__init__(*args)

        # self.xk_1 # state vector mean at time step k-1 inherited from FEKFMBL

        self.nzm = 0  # number of measurements observed
        self.nzf = 0  # number of features observed

        self.H = None  # Data Association Hypothesis
        self.nf = 0  # number of features in the state vector

        self.plt_MappedFeaturesEllipses = []

        return

    def AddNewFeatures(self, xk, Pk, znp, Rnp):
        """
        This method adds new features to the map. Given:

        * The SLAM state vector mean and covariance:

        .. math::
            {{}^Nx_{k}} & \\approx {\\mathcal{N}({}^N\\hat x_{k},{}^NP_{k})}\\\\
            {{}^N\\hat x_{k}} &=   \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k}}&=
            \\begin{bmatrix}
            {{}^NP_{B}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance

        * And the vector of non-paired feature observations (feature which have not been associated with any feature in the map), and their covariance matrix:

            .. math::
                {z_{np}} &=   \\left[ {}^Bz_{F_1} ~  \\cdots ~ {}^Bz_{F_{n_{zf}}}  \\right]^T \\\\
                {R_{np}}&= \\begin{bmatrix}
                {}^BR_{F_1} &  \\cdots & 0  \\\\
                \\vdots &  \\ddots & \\vdots \\\\
                0 & \\cdots & {}^BR_{F_{n_{zf}}}
                \\end{bmatrix}
                :label: FEKFSLAM-non-paire-feature-observations

        this method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
        Therefore, for each new feature :math:`{}^Bz_{F_i}`, included in the vector :math:`z_{np}`, and its corresponding feature observation noise :math:`{}^B R_{F_i}`, the state vector mean and covariance are updated as follows:

            .. math::
                {{}^Nx_{k}^+} & \\approx {\\mathcal{N}({}^N\\hat x_{k}^+,{}^NP_{k}^+)}\\\\
                {{}^N x_{k}^+} &=
                \\left[ {{}^N x_{B_k}^T} ~ {{}^N x_{F_1}^T} ~ \\cdots ~{{}^N x_{F_n}^T}~ |~\\left({{}^N x_{B_k} \\boxplus ({}^Bz_{F_i} }+v_k)\\right)^T \\right]^T \\\\
                {{}^N\\hat x_{k}^+} &=
                \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~ \\cdots ~{{}^N\\hat x_{F_n}^T}~ |~{{}^N\\hat x_{B_k} \\boxplus {}^Bz_{F_i}^T } \\right]^T \\\\
                {P_{k}^+}&= \\begin{bmatrix}
                {{}^NP_{B_k}}  &  {{}^NP_{B_kF_1}}   &  \\cdots   &  {{}^NP_{B_kF_n}} & | & {{}^NP_{B_k} J_{1 \\boxplus}^T}\\\\
                {{}^NP_{F_1B_k}}  &  {{}^NP_{F_1}}   &  \\cdots   &  {{}^NP_{F_1F_n}} & | & {{}^NP_{F_1B_k} J_{1 \\boxplus}^T}\\\\
                \\vdots  & \\vdots & \\ddots  & \\vdots & | &  \\vdots \\\\
                {{}^NP_{F_nB_k}}  &  {{}^NP_{F_nF_1}}   &  \\cdots   &  {{}^NP_{F_n}}  & | & {{}^NP_{F_nB_k} J_{1 \\boxplus}^T}\\\\
                \\hline
                {J_{1 \\boxplus} {}^NP_{B_k}}  &  {J_{1 \\boxplus} {}^NP_{B_kF_1}}   &  \\cdots   &  {J_{1 \\boxplus} {}^NP_{B_kF_n}}  & | &  {J_{1 \\boxplus} {}^NP_R J_{1 \\boxplus} ^T} + {J_{2\\boxplus}} {{}^BR_{F_i}} {J_{2\\boxplus}^T}\\\\
                \\end{bmatrix}
                :label: FEKFSLAM-add-a-new-feature

        :param xk: state vector mean
        :param Pk: state vector covariance
        :param znp: vector of non-paired feature observations (they have not been associated with any feature in the map)
        :param Rnp: Matrix of non-paired feature observation covariances
        :return: [xk_plus, Pk_plus] state vector mean and covariance after adding the new features
        """

        #assert znp.size > 0, "AddNewFeatures: znp is empty"

        
        ## To be completed by the student

        # xk the current state vector mean with robot pose and already mapped features
        # Pk the current state vector covariance with robot pose and already mapped features
        # znp the vector of non-paired feature observations, these are measuremenrs of fetaires that have not yet been associated with any known feature in the map
        # Rnp the covariance matrix of the non-paired feature observations

        # for each new feature observation zfi in B frame in znp :

        # 1. add the new feature position in N frame to the state vector mean , it is calculated as N`xfi = N`xBk (+)  B`zfi 
        # 2. add the new feature position in N frame to the state vector covariance, it is calculated as Pk_plus = [Pkk Pkf; Pfk Pff] where Pkf = Pk * J1x.T and Pff = J1x * Pk * J1x.T + J2x * R * J2x.T
        # 3. create a grown state vector xk_plus and Pk_plus

        xk_plus = xk   
        Pk_plus = np.array(Pk)

        zfi_dim = self.xF_dim  # dimension of the new feature
        xB_dim = self.xB_dim  # dimension of the robot pose

        xBk = Pose3D(xk[0:xB_dim]) # robot pose in the N frame
        PBk = Pk[0:xB_dim, 0:xB_dim] # covariance of the robot pose
        
        #nf = len(znp) // zfi_dim  # number of new features
        #for i in range(nf):
        
        for i in range(len(znp)):

            #zfi = znp[i]
            #Rfi = Rnp[i]

            # calculate the new feature position in the N frame
            xfi = self.g(xBk, znp[i])

            # append the new feature to the state vector mean
            xk_plus = np.vstack([xk_plus, xfi])
       
            # Pk_plus = G1 * Pk_1 * G1.T + G2 * Rk * G2.T
            # Pk_plus = [Pk Pcross.T ; Pcross Pnew]

            # calculate the Jacobians
            Jgxi = self.Jgx(xBk, znp[i])
            Jgvi = self.Jgv(xBk, znp[i])
            
            # compute the covariance of the new feature
            Pfi = Jgxi @ PBk @ Jgxi.T + Jgvi @ Rnp[i] @ Jgvi.T

            # compute the cross-covariance between the robot's pose and the new feature
            Pk_B = Pk[0:xB_dim, 0:xB_dim]       # covariance of robot's pose with itself
            Pk_BF = Pk[0:xB_dim, xB_dim:]       # cross-covariance with existing features
            Pcross =  np.block([Jgxi @ Pk_B,Jgxi @ Pk_BF])
       
            # update the state vector covariance
            Pk_plus = np.block([[ Pk_plus, Pcross.T], [Pcross, Pfi]])
            Pk = Pk_plus
            self.nf +=1 # increment the number of features

        return xk_plus, Pk_plus


        # return xk_plus, Pk_plus  , the state vector mean and covariance after adding the new features

    def Prediction(self, uk, Qk, xk_1, Pk_1):
        """
        This method implements the prediction step of the FEKFSLAM algorithm. It predicts the state vector mean and
        covariance at the next time step. Given state vector mean and covariance at time step k-1:

        .. math::
            {}^Nx_{k-1} & \\approx {\\mathcal{N}({}^N\\hat x_{k-1},{}^NP_{k-1})}\\\\
            {{}^N\\hat x_{k-1}} &=   \\left[ {{}^N\\hat x_{B_{k-1}}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k-1}}&=
            \\begin{bmatrix}
            {{}^NP_{B_{k-1}}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance-k-1

        the control input and its covariance :math:`u_k` and :math:`Q_k`, the method computes the state vector mean and covariance at time step k:

        .. math::
            {{}^N\\hat{\\bar x}_{k}} &=   \\left[ {f} \\left( {{}^N\\hat{x}_{B_{k-1}}}, {u_{k}}  \\right)  ~  { {}^N\\hat x_{F_1}^T} \\cdots { {}^N\\hat x_{F_n}^T}\\right]^T\\\\
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}
            :label: FEKFSLAM-prediction-step

        where

        .. math::
            {F_{1_k}} &= \\left.\\frac{\\partial {f_S({}^Nx_{k-1},u_k,w_k)}}{\\partial {{}^Nx_{k-1}}}\\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}} \\\\
             &=
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{B_{k-1}}}} &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{Fn}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}} \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_x}} & {0} & \\cdots & {0} \\\\
            {0}   & {I} & \\cdots & {0} \\\\
            \\vdots& \\vdots  & \\ddots & \\vdots  \\\\
            {0}   & {0} & \\cdots & {I} \\\\
            \\end{bmatrix}
            \\\\{F_{2_k}} &= \\left. \\frac{\\partial {f({}^Nx_{k-1},u_k,w_k)}}{\\partial {w_{k}}} \\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}}
            =
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {w_{k}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {w_{k}}}\\\\
            \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {w_{k}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_w}}\\\\
            {0}\\\\
            \\vdots\\\\
            {0}\\\\
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-Jacobian

        obtaining the following covariance matrix:   update the correlation between the robot and the map
        
        .. math::
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}{{}^N\\bar P_{k}}
             &=
            \\begin{bmatrix}
            {J_{f_x}P_{B_{k-1}} J_{f_x}^T} + {J_{f_w}Q J_{f_w}^T}  & |  &  {J_{f_x}P_{B_kF_1}} & \\cdots & {J_{f_x}P_{B_kF_n}}\\\\
            \\hline
            {{}^NP_{F_1B_k} J_{f_x}^T} & |  &  {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_n}}\\\\
            \\vdots & | & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_nB_k} J_{f_x}^T} & | &  {{}^NP_{F_nF_1}} & \\cdots & {{}^NP_{F_n}}
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-covariance

        The method returns the predicted state vector mean (:math:`{}^N\\hat{\\bar x}_k`) and covariance (:math:`{{}^N\\bar P_{k}}`).

        :param uk: Control input
        :param Qk: Covariance of the Motion Model noise
        :param xk_1: State vector mean at time step k-1
        :param Pk_1: Covariance of the state vector at time step k-1
        :return: [xk_bar, Pk_bar] predicted state vector mean and covariance at time step k
        """

        ## To be completed by the student
        # predict the state vector mean and covariance at time step k
        # we have xBk_1, xFi and Pk_1 , we calculate xk_bar and Pk_bar while xFi remained unchanged

        self.xk_1 = xk_1
        self.Pk_1 = Pk_1
        self.uk = uk
        self.Qk = Qk    
        
        # xk_bar = [f(xk_1, uk, 0) xF1 ..... xFn]T
        xk_bar = np.array(xk_1)
        xk_bar[0:3] = self.f(xk_1[0:3], uk)

        xBk_1 = Pose3D(xk_1[0:3])

        # Covariance Prediction: Pk_bar = F1*Pk_1*F1.T + F2*Qk*F2.T
        # Covariance Matrix: Pk_bar = np.block([ Pxx, Pxf], [Pfx, Pff])
        # Pxx is the covariance of the robot pose
        J_fx = self.Jfx(xBk_1)
        J_fw = self.Jfw(xBk_1)

        NPk = Pk_1[0:3,0:3]
        Pxx = J_fx @ NPk @ J_fx.T + J_fw @ Qk @ J_fw.T
        # Pxf is the correlation between the robot and the features
        Pxf = J_fx @ Pk_1[0:3,3:]
        # Pfx is the correlation between the features and the robot
        Pfx = Pxf.T
        # Pff is the covariance of the features
        Pff = Pk_1[3:,3:]
        Pk_bar = np.block([[Pxx,Pxf], [Pfx,Pff]])

        return xk_bar, Pk_bar
        

    def Localize(self, xk_1, Pk_1):
        """
        This method implements the FEKFSLAM algorithm. It localizes the robot and maps the features in the environment.
        It implements a single interation of the SLAM algorithm, given the current state vector mean and covariance.
        The unique difference it has with respect to its ancestor :meth:`FEKFMBL.Localize` is that it calls the method
        :meth:`AddNewFeatures` to add new non-paired features to the map.

        :param xk_1: state vector mean at time step k-1
        :param Pk_1: covariance of the state vector at time step k-1
        :return: [xk, Pk] state vector mean and covariance at time step k
        """

        ## To be completed by the student
        uk, Qk = self.GetInput()
        xk_bar, Pk_bar  = self.Prediction(uk, Qk, xk_1, Pk_1)
        zm, Rm, Hm, Vm  = self.GetMeasurements()
        zf, Rf  = self.GetFeatures()
        Hp = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)
        [zk, Rk, Hk, Vk, znp, Rnp] = self.StackMeasurementsAndFeatures(xk_bar, zm, Rm, Hm, Vm, zf, Rf, Hp)
        xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)
        
        if(len(znp) >= self.xF_dim):
             xk,Pk = self.AddNewFeatures(xk_bar, Pk_bar,znp,Rnp)
    
        self.xk, self.Pk = xk, Pk
        # Use the variable names zm, zf, Rf, znp, Rnp so that the plotting functions work
        self.Log(self.robot.xsk, self.GetRobotPose(self.xk), self.GetRobotPoseCovariance(self.Pk),
                 self.GetRobotPose(self.xk), zm)  # log the results for plotting
     
        return self.xk, self.Pk, zf, Rf, znp, Rnp

    def LocalizationLoop(self, x0, P0, usk):
    
        xk_1 = x0
        Pk_1 = P0

        xsk_1 = self.robot.xsk_1

        zf, Rf = self.GetFeatures()

        # add new features to the state vector and covariance matrix
        xk_1, Pk_1 = self.AddNewFeatures(xk_1, Pk_1, zf, Rf)

        for self.k in range(self.kSteps):

            xsk = self.robot.fs(xsk_1, usk)  # simulate the robot motion
            xk, Pk, zf, Rf, znp, Rnp = self.Localize(xk_1, Pk_1)  # localize the robot
            zf = np.array(zf).reshape(len(zf) * self.xF_dim ,1) # zf becomes fJ rows and 1 column
            Rf = block_diag(*Rf) # take Rf and pass its elements as arg to block_diag to craete a block diagonal matrix 
            self.PlotUncertainty(zf, Rf, znp, Rnp)

            xsk_1 = xsk  # current state becomes previous state for next iteration
            xk_1 = xk
            Pk_1 = Pk

        self.PlotState()  # plot the state estimation results
        plt.show()

        
    def PlotMappedFeaturesUncertainty(self):
        """
        This method plots the uncertainty of the mapped features. It plots the uncertainty ellipses of the mapped
        features in the environment. It is called at each Localization iteration.
        """
        # remove previous ellipses
        for i in range(len(self.plt_MappedFeaturesEllipses)):
            self.plt_MappedFeaturesEllipses[i].remove()
        self.plt_MappedFeaturesEllipses = []

        self.xk=BlockArray(self.xk,self.xF_dim, self.xB_dim)
        self.Pk=BlockArray(self.Pk,self.xF_dim, self.xB_dim)

        # draw new ellipses
        for Fj in range(self.nf):
            feature_ellipse = GetEllipse(self.xk[[Fj]],
                                         self.Pk[[Fj,Fj]])  # get the ellipse of the feature (x_Fj,P_Fj)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], 'r')  # plot it
            self.plt_MappedFeaturesEllipses.append(plt_ellipse)  # and add it to the list

    def PlotUncertainty(self, zf, Rf, znp, Rnp):
        """
        This method plots the uncertainty of the robot (blue), the mapped features (red), the expected feature observations (black) and the feature observations.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param znp: vector of non-paired feature observations
        :param Rnp: covariance matrix of non-paired feature observations
        :return:
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(znp, Rnp,'b')
            self.PlotFeatureObservationUncertainty(zf, Rf,'g')
            self.PlotExpectedFeaturesObservationsUncertainty()
            self.PlotMappedFeaturesUncertainty()
