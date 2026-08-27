function [obj] = ONMF(X)
%%X:特征矩阵 Y:标签矩阵 R:特征相似度矩阵（局部图连接） A:样本相似度矩阵（近邻） alpha,beta,gamma:平衡参数
%%k:中间维度（Z的中间维度）
[n_sam,n_fea]= size(X);
k = n_fea;
A = generateNOM(n_sam,k);

S = rand(n_fea, k);

iter = 1; obji = 1;
while iter<201
    %%更新A

    AM = X*S';
    AZ = A*A'*AM;
    AZ(AZ == 0) = eps;
    AZ1 = AM./(AZ);
    AZ2 = sqrt(AZ1);
    A = A.*AZ2;
    
    %%更新S
    SM = A'*X;
    SZ = A'*A*S;
    SZ(SZ == 0) = eps;
    SZ1 = SM./(SZ);
    S = S.*SZ1;
    
    
    obj(iter)=norm((X-A*S),'fro')^2;
    

end
end
