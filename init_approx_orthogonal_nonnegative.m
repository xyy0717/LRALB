function A = init_approx_orthogonal_nonnegative(n, k, sparsity)
% 初始化一个稀疏、非负、列近似正交的矩阵 (n x k)
% 输入:
%   n        - 行数
%   k        - 列数 (要求 k <= n)
%   sparsity - 稀疏比例，例如 0.1 表示约 10% 的元素非零
% 输出:
%   A        - n x k 非负矩阵，每列 L2 范数为 1

    if nargin < 3
        sparsity = 0.5;  % 默认稀疏度
    end

    % 步骤1: 生成 [0,1) 均匀随机矩阵
    A = rand(n, k);

    % 步骤2: 生成稀疏掩码（逻辑矩阵）
    mask = rand(n, k) < sparsity;

    % 步骤3: 应用掩码，得到稀疏非负矩阵
    A = A .* mask;

    % 步骤4: 对每一列进行 L2 归一化（避免全零列）
    col_norms = sqrt(sum(A.^2, 1));  % 每列的 L2 范数 (1 x k)
    
    % 处理可能的全零列（虽然概率低，但安全起见）
    col_norms(col_norms == 0) = 1;  % 避免除零；实际中可重新采样或设为小常数
    
    A = A ./ col_norms;  % 广播除法：每列除以其范数
end