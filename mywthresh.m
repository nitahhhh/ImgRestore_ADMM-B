function y=mywthresh(x,sorh,t)
%avoid no wavelet licsens 
switch sorh
    case 's'
        tmp=(abs(x)-t);
        %tmp=(tmp+abs(tmp))/2;
        tmp=max(tmp,0);
        y=sign(x).*tmp;
    case 'h'
        y=x.*(abs(x)>t);
    otherwise
        error('invalid argument value');
end