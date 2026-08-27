function W = Laplacian_GK(X, k)
% each column is a data
% 输入: X (nFea × nSmp), k: 近邻数
% 输出: W (nSmp × nSmp 稀疏对称)

[~, nSmp] = size(X);

%% 1. 平方欧氏距离
Dsq = pdist2(X', X', 'squaredeuclidean');

%% 2. 部分排序 —— 只取 k+1 个最近邻
try
    [dsq_nn, idx_nn] = mink(Dsq, k+1, 2);        % R2017b+
catch
    [dsq_sorted, idx_sorted] = sort(Dsq, 2);
    dsq_nn = dsq_sorted(:, 1:k+1);
    idx_nn = idx_sorted(:, 1:k+1);
end
clear Dsq;

% 去掉自身
dsq_knn = dsq_nn(:, 2:k+1);     % n×k 平方距离
idx_knn = idx_nn(:, 2:k+1);     % n×k 近邻索引
clear dsq_nn idx_nn;

%% 3. sigma —— 向量化（替代 n×n 的 H 矩阵）
sigma = sum(sqrt(dsq_knn(:))) / nSmp;

%% 4. 高斯核权重 —— 向量化（替代双重for循环）
weights = exp(-dsq_knn / (2 * sigma^2));

%% 5. 构建稀疏矩阵 + 对称化
rows = repmat((1:nSmp)', 1, k);
W = sparse(rows(:), idx_knn(:), weights(:), nSmp, nSmp);
W = max(W, W');    % 对称化，与原始 max(W,W') 完全一致

end