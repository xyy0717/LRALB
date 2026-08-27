function [A, beta] = ConstructA_Lcoal(TrainData, k)
% 输入: TrainData (n × d), k: 近邻数
% 输出: A (n × n 稀疏), beta: 参数
% 接口完全不变，主函数无需任何修改

[n, ~] = size(TrainData);
kk = k + 1;  % 需要 k+1 个近邻（不含自身）

%% ====== 1. 距离计算: pdist2 内置BLAS优化 ======
% 原始: Dis = EuDist2(TrainData', TrainData', 0);  % 自定义函数
Dis = pdist2(TrainData, TrainData, 'squaredeuclidean');  % 等价于 EuDist2(...,0)

%% ====== 2. 部分排序: mink 只找最小 k+2 个 ======
% 原始: [~,idx] = sort(Dis, 2);  % 完整排序 O(n·log n) per row
% 优化: mink 部分排序 O(n) per row（R2017b+）
[di_all, idx_all] = mink(Dis, k + 2, 2);  % n × (k+2)
clear Dis;  % 立即释放 n×n 矩阵（~301MB）

% 去掉自身（第1列，距离=0）
di_sorted = di_all(:, 2:k+2);    % n × (k+1)
idx_nn    = idx_all(:, 2:k+2);   % n × (k+1)
clear di_all idx_all;

%% ====== 3. 向量化权重计算（消除for循环）======
% 原始: for i = 1:num1 逐行计算
di_k1  = di_sorted(:, kk);               % 第 k+1 近邻距离, n×1
sum_dk = sum(di_sorted(:, 1:k), 2);       % 前 k 个距离之和, n×1
denom  = k * di_k1 - sum_dk + eps;        % 分母, n×1

% 广播计算所有权重: (di_k1 - di_j) / denom
weights = (di_k1 - di_sorted) ./ denom;   % n × (k+1)

% 处理所有距离相等的情况
equal_mask = (denom <= eps);               % 逻辑向量 n×1
if any(equal_mask)
    weights(equal_mask, :)   = 0;
    weights(equal_mask, 1:k) = 1 / k;
end

%% ====== 4. 构建稀疏矩阵（替代稠密 n×n 矩阵）======
% 原始: A = zeros(num1, num2);  % 301MB 稠密矩阵
% 优化: 稀疏矩阵，仅 ~1.2MB
rows = repmat((1:n)', 1, kk);             % n × (k+1)
A = sparse(rows(:), idx_nn(:), weights(:), n, n);  % n × n 稀疏

%% ====== 5. 计算 beta ======
bb = k/2 * di_k1 - 0.5 * sum_dk;
bb(equal_mask) = 0;
beta = mean(bb);

end