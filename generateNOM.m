function Q = generateNOM(n, k, max_iter)
% 快速生成一个 n × k 的非负、稠密、近似列正交的矩阵 Q（交替投影法）
% max_iter：迭代次数，默认100

    if nargin < 3
        max_iter = 100;
    end

    Q = rand(n, k);  % 初始非负稠密矩阵

    for iter = 1:max_iter
        % Step 1: 归一化每列
        Q = Q ./ (vecnorm(Q) + 1e-8);  % 防止除以0

        % Step 2: 施加列正交：用 QR 投影逼近列正交
        [Q_orth, ~] = qr(Q, 0);  % Q_orth 是列正交的，size n × k

        % Step 3: 非负修正（投影到非负空间）
        Q = max(Q_orth, 0);

        % Step 4: 可选 - 归一化列（保持数值稳定）
%         Q = Q ./ (vecnorm(Q) + 1e-8);
    end
end