function C = colCosineSim(X)
% colCosineSim 计算矩阵列向量的余弦相似度矩阵
%   输入:
%       X - m x n 矩阵，每列为一个向量
%   输出:
%       C - n x n 余弦相似度矩阵，对角线为 0，半正定

    % 1. 每列归一化
    norms = sqrt(sum(X.^2, 1));   % 列范数
    X_norm = X ./ norms;          % 归一化列向量

    % 2. 计算余弦相似度矩阵
    C = X_norm' * X_norm;         % C(i,j) = cos(X(:,i), X(:,j))



    % 4. 保证对称性（数值误差处理）
    C = (C + C') / 2;
end
