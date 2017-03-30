%ADMM-B
clear all;
close all;
clc;
sigma=3; %standard deviation of gaussian blur kernel 
dim=9; %blur kernel window size
%deconv parameters
lambda=10^-10; %deblur without noise
%lambda=0.006; %deblur with noise
add_noise=0; %0:no noise,1:add noise
method=1; %0: deblur; 1:inpainting
kernel_mode=0; % 0:gaussian; 1:uniform
admm_mode=0; %0:ADMM-B,1:ADMM-S,2:ADMM-A
paint_mode=1; % 0:random hole; 1:shape hole
if (kernel_mode==0)
    H = fspecial('gaussian', dim,sigma); %gaussian blur
elseif(kernel_mode==1)
    H= ones(dim)/dim/dim;
end
if (admm_mode==0)
    gamma=1;%admm-b
    name='ADMM-B';
elseif(admm_mode==1)
    gamma=0; %admm-s
    name='ADMM-S';
else
    gamma=10000000000; %admm-a
    name='ADMM-A';
end
%%%notation
%blur_image: Y
%good_image: X
%blur_kernel: H

load=double(imread('shot.png'));
X=imresize(load(:,:,1),1);
Y=X; %just declare Y
if(~method) %create blurred image
    Y=imfilter(X,H,'circular'); %use circular! or, edge ringing in deblur image
else %create painted image
    %%%%%%%%%%Random holes%%%%%%%%%%
	if (~paint_mode)
		[m,n,c]=size(load);
		P =random('norm',2,0.3,m,n);
		P(P<1.5)=0;
		P(P>=1.5)=1;
	%%%%%%%%%%shape missing holes%%%%%%%%%%
    else
		P =rgb2gray(double(imread('mask_thick.png'))); 
	end
	Y=P.*Y; 
end
if(add_noise==1)
    Y=Y+3*rand(size(Y)); %add noise
end
y=reshape(Y,[],1);  %column vector
figure; imagesc(Y); colormap(gray); title('blur img');
%%
%get size
[Hrow,Hcol]=size(H);
[Xrow,Xcol]=size(X);
[Yrow,Ycol]=size(Y);
n=Xrow*Xcol;
m=Yrow*Ycol;

%define wavelet transform
frame = 1 ; Level =  4; %choose haar filter and 4 level decomposition
[D,R]=GenerateFrameletFilter(frame); %get filter
nD=length(D); 
dL=9*Level; 
len=3;
if(~frame) len=2; end
W  = @(x) FraDecMultiLevel(x,D,Level); % Frame decomposition
WT = @(x) FraRecMultiLevel(x,R,Level); % Frame reconstruction
% Note different notation from paper:  W in paper => WT(); WT in paper => W();

%%%other params.
mu=0.1*lambda;
alpha=mu/(mu+gamma);
%%%
%precompute something
%blur matrix in f domain
if(~method)
	Mask=zeros(size(X));
	Mask([end+1-floor(Hrow/2):end,1:ceil(Hrow/2)],[end+1-floor(Hcol/2):end,1:ceil(Hcol/2)]) = H;
	FMask=fft2(Mask);%fft2(H) with padding to image size
	FW=(mu+abs(FMask).^2).^-1;
end
%%%

k=1; %start iteration
%Tol: error tolerance
if(~method) %deblur
    Tol=10;
else %inpainting
    Tol=0.00002;
end
%initialize
x=W(Y); % x v d r are coeff in wavelet domain, guess x as observation of blur image
v=x;
d=x;
r=x;
terminate=0;

if(~method) %deblur
WtBty=W(real(ifft2(conj(FMask).*fft2(Y)))); 
else %inpainting
    WtBty=W(conj(P).*Y);
end
PSNR_mat=zeros();
mse_mat=zeros();
while(~terminate)
    %update r
    vpd=CoeffOper('+',v,d); 
    r=CoeffOper('+',WtBty,CoeffOper('*c',vpd,mu));
    
    %update x
    WT_r=WT(r); 
    WT_r(WT_r>255)=255; WT_r(WT_r<0)=0;
    
    if(~method) %deblur
		beta=W(real(ifft2( conj(FMask).*FW.*FMask.*fft2(WT_r)))); 
    else %inpainting
        beta=W( 1/(mu+1)*(abs(P).^2).*WT_r); 
    end
    apart=CoeffOper('+',CoeffOper('*c',r,alpha),CoeffOper('*c',W(WT_r),1-alpha));
	x=CoeffOper('*c',(CoeffOper('-',apart,beta)), 1/mu);
    
    %update v
	v_p=CoeffOper('-',x,d); 
    v=CoeffOper('sc',v_p, lambda/mu); %soft thresh
	%update d
    d=CoeffOper('-',d,CoeffOper('-',x,v)); 
    
	
	%visualize result   
	U=WT(x);  
    U(U>255)=255; U(U<0)=0;
	%figure; imagesc(U); colormap(gray); title('deblur');
	%stop criterion (have not written)
    if(~mod(k,2)) %show some median result
        figure; imagesc(U); colormap(gray); title(['deblur iter=',num2str(k),'times']);
    end
    %PSNR and MSE
    A=load(:,:,1);
    B=U;
    dif = sum((A(:)-B(:)).^2) / numel(A); 
    PSNR = 10*log10(255*255/dif)
    mse = mean2((A-B).^2)
    PSNR_mat(k)=PSNR;
    mse_mat(k)=mse;
	if(k>=2)
        obj=abs(U-(U_last));
        U_last2=(U_last==0)*255+U_last;%avoid divided by zero
        if (sum(sum(obj./(U_last2))) <Tol)||(k>=40)
            terminate=1;
            figure; imagesc(U); colormap(gray); title([num2str(name),' , PSNR=',num2str(PSNR),' , MSE=',num2str(mse),' , lambda=',num2str(lambda)]);
        end       
    end
    k=k+1;
    U_last=U;
end
imwrite(uint8(U_last),'Deblur_Result.png')
   
    figure(100),plot(PSNR_mat,'-','linewidth' ,4) ;title(['PSNR(',num2str(name),' , lambda=',num2str(lambda)],'fontsize',14);
    xlabel('iteration','fontsize',14)
    ylabel('PSNR','fontsize',14)
    figure(200),plot(mse_mat,'-','linewidth' , 4) ;title(['MSE(',num2str(name),' , lambda=',num2str(lambda)],'fontsize',14);
    xlabel('iteration','fontsize',14)
    ylabel('MSE','fontsize',14)

