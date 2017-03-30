function Dec=FraDecMultiLevel(A,D,L)

% function Dec=FraDecMultiLevel(A,D,L)
% This function implement framelet decomposition up to level L.
% A ==== the data to be decomposed, which are in a square matrix.
% D ==== is the decomposition filter in 1D. In 2D, it is generated by tensor
% product. The filter D must be symmetric or anti-symmetric, which
% are indicated by 's' and 'a' respectively in the last cell of D.
% L ==== is the level of the decomposition.
% Dec ==== is the framlet coefficient which are in a cell.

% Written by Jian-Feng Cai.
% email: tslcaij@nus.edu.sg

nD=length(D);

kDec=A;
for k=1:L
    Dec{k}=FraDec(kDec,D,k);
    kDec=Dec{k}{1,1};
end

% [m,n]=size(A);
% kDec=A;
% Dec=[];
% SorAS=D{nD};
% for k=1:L
%     for i=nD-1:-1:1
%         M1=D{i};
%         for j=nD-1:-1:1
%             M2=D{j};
%             temp=ConvSymAsym(kDec,M1,SorAS(i),L);
%             temp=ConvSymAsym(temp',M2,SorAS(j),L);
%             temp=temp';
%             if (i==1)&(j==1)
%                 kDec=temp;
%             else
%                 Dec=[Dec;temp];
%             end
%         end
%     end
% end
% Dec=[Dec;kDec];