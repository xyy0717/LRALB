function [W1, obj] = LRALB(X, Y, R, A, alpha, beta, gamma, k, precomp)
%% X:特征矩阵 Y:标签矩阵 R:特征相似度矩阵 A:样本相似度矩阵
%% alpha,beta,gamma:平衡参数  k:中间维度
%% precomp: 预计算结构体（可选），包含 LA, DA_vec, lambda_max, R, DR_vec

[n_sam, n_fea] = size(X);
n_lab = size(Y, 2);
rng(42)

%% ====== 使用预计算数据或现场计算 ======
if nargin >= 9 && ~isempty(precomp)
    % 直接使用预计算结果，跳过耗时计算
    LA         = precomp.LA;
    DA_vec     = precomp.DA_vec;
    lambda_max = precomp.lambda_max;
    R          = precomp.R;
    DR_vec     = precomp.DR_vec;
else
    % 兼容旧调用方式：现场计算
    DA_vec     = full(sum(A, 2));
    LA         = spdiags(DA_vec, 0, n_sam, n_sam) - A;
    R          = (R + R') / 2;
    R(1:size(R,1)+1:end) = 0;
    DR_vec     = full(sum(R, 2));
    if issparse(LA) && n_sam > 50
        lambda_max = eigs(LA, 1, 'largestabs');
    else
        lambda_max = max(eig(full(LA)));
    end
end

%% 初始化
Z     = init_approx_orthogonal_nonnegative(n_sam, k);
H     = rand(n_fea, k);
W     = rand(k, n_lab);
C     = zeros(n_sam, n_lab);
theta = 0;
W1    = H * W;

obj   = zeros(1, 100);  % 预分配
iter  = 1;
obji  = 1;

while iter < 101
    %% 更新 mu
    mu = 1 / (theta + beta * lambda_max);

    %% 更新 hatY
    hatY = theta * C + (1 - theta) * Y;

    %% 更新 Z —— 关键优化: Z*(Z'*ZM) 替代 (Z*Z')*ZM
    ZM = X * H + hatY * W';
    ZZ = Z * (Z' * ZM);               % k×k 中间矩阵，O(nk²) 替代 O(n²k)
    ZZ = max(ZZ, eps);
    Z  = Z .* sqrt(ZM ./ ZZ);

    %% 更新 H —— 优化: DR_vec.*H 替代 diag(DR)*H; d.*(...) 替代 diag(d)*(...)
    XtZ  = X' * Z;                     % n_fea × k
    d    = 0.5 ./ sqrt(sum(W1.^2, 2) + eps);  % n_fea × 1
    RH   = R * H;                      % 稀疏乘法

    HM   = XtZ + alpha * RH;
    HWWt = (H * W) * W';              % n_fea × k
    HZ   = H + alpha * (DR_vec .* H) + gamma * (d .* HWWt);
    HZ   = max(HZ, eps);
    H    = H .* (HM ./ HZ);

    %% 更新 W —— 优化: d.*H 替代 diag(d)*H
    ZtZ  = Z' * Z;                     % k × k
    HtDH = H' * (d .* H);             % k × k

    WM = Z' * hatY;
    WZ = ZtZ * W + gamma * (HtDH * W);
    WZ = max(WZ, eps);
    W  = W .* (WM ./ WZ);

    %% 更新 C —— 优化: DA_vec.*C 替代 diag(DA)*C
    ZW      = Z * W;
    LAC     = DA_vec .* C - A * C;     % LA*C，避免显式构建大矩阵
    Delta_C = 2 * (hatY - ZW) + 2 * beta * LAC;
    C       = min(max(C - mu * Delta_C, 1), Y);

    %% 更新 theta
    thetaA = ZW - Y;
    thetaB = C - Y;
    thetaM = thetaA(:)' * thetaB(:);
    thetaZ = max(thetaB(:)' * thetaB(:), eps);
    theta  = min(max(thetaM / thetaZ, 0), 1);

    %% 更新 W1
    W1 = H * W;

    %% 计算目标函数（trace展开，避免大矩阵）
    HtH   = H' * H;
    term1 = sum(X(:).^2) - 2*sum(XtZ(:).*H(:)) + sum(ZtZ(:).*HtH(:));
    diff2 = ZW - theta*C - (1-theta)*Y;
    term2 = sum(diff2(:).^2);
    term3 = alpha * (sum(DR_vec .* sum(H.^2,2)) - sum(H(:).*RH(:)));
    AC    = A * C;
    term4 = beta * (sum(DA_vec .* sum(C.^2,2)) - sum(C(:).*AC(:)));
    term5 = gamma * sum(sqrt(sum(W1.^2, 2) + eps));

    obj(iter) = term1 + term2 + term3 + term4 + term5;

    cver = abs((obj(iter) - obji) / obji);
    obji = obj(iter);
    iter = iter + 1;
    if cver < 1e-3 && iter > 3
        break
    end
end

obj = obj(1:iter-1);
end
