clc; clear;
addpath(genpath(pwd));


load("Medical0.5.mat")

nfold = 5;
data = columnMinMaxNormalization(data);
[N, num_feature] = size(data);

if num_feature <= 100,       Theta = 0.4;
elseif num_feature <= 500,   Theta = 0.3;
elseif num_feature <= 1000,  Theta = 0.2;
else,                        Theta = 0.1;
end

list  = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
list1 = [0.1, 0.2, 0.3, 0.4];
ACRmax   = 0;
k_select = fix(Theta * num_feature);

%% ===============================================================
%  步骤1: 固定交叉验证划分
%  ===============================================================
rng(42);
indices_all = crossvalind('Kfold', 1:N, nfold);  % 只生成一次

%% ===============================================================
%  步骤2: 预计算每折的数据、相似度矩阵和 precomp 结构体
%  ===============================================================


fold_train_data        = cell(nfold, 1);
fold_test_data         = cell(nfold, 1);
fold_S_A               = cell(nfold, 1);
fold_S_R               = cell(nfold, 1);
fold_precomp           = cell(nfold, 1);    % ← 预计算结构体
fold_train_target_pmfs = cell(nfold, 1);    % PMFS用 (0/1)
fold_train_target_ml   = cell(nfold, 1);    % MLKNN用 (-1/1)
fold_test_target_ml    = cell(nfold, 1);

for i = 1:nfold
    tic_fold = tic;
    test_idxs  = (indices_all == i);
    train_idxs = ~test_idxs;

    fold_train_data{i} = data(train_idxs, :);
    fold_test_data{i}  = data(test_idxs, :);

    tr_target = candidate_labels(:, train_idxs);
    te_target = target(:, test_idxs);

    % PMFS标签 (0/1)
    tmp_pmfs = tr_target;
    tmp_pmfs(tmp_pmfs == -1) = 0;
    fold_train_target_pmfs{i} = tmp_pmfs';   % n_train × n_lab

    % MLKNN标签 (-1/1)
    tmp_ml = tr_target;
    tmp_ml(tmp_ml == 0) = -1;
    fold_train_target_ml{i} = tmp_ml;

    tmp_te = te_target;
    tmp_te(tmp_te == 0) = -1;
    fold_test_target_ml{i} = tmp_te;

    % ---- 计算相似度矩阵（每折仅一次）----
    S_A = Laplacian_GK(fold_train_data{i}', 10);
    [S_R, ~] = ConstructA_Lcoal(fold_train_data{i}', 10);

    fold_S_A{i} = S_A;
    fold_S_R{i} = S_R;

    % ---- 构建 precomp 结构体 ----
    pc = struct();
    DA_vec = full(sum(S_A, 2));
    n_tr   = size(S_A, 1);
    pc.LA  = spdiags(DA_vec, 0, n_tr, n_tr) - S_A;
    pc.DA_vec = DA_vec;

    % 只求最大特征值
    if issparse(pc.LA) && n_tr > 50
        pc.lambda_max = eigs(pc.LA, 1, 'largestabs');
    else
        pc.lambda_max = max(eig(full(pc.LA)));
    end

    % R 预处理（对称化 + 零对角）
    R_sym = (S_R + S_R') / 2;
    R_sym(1:size(R_sym,1)+1:end) = 0;
    pc.R      = R_sym;
    pc.DR_vec = full(sum(R_sym, 2));

    fold_precomp{i} = pc;

    fprintf('  第 %d/%d 折预计算完成，耗时 %.1f 秒\n', i, nfold, toc(tic_fold));
end
fprintf('========== 预计算完成 ==========\n\n');

%% ===============================================================
%  步骤3: 网格搜索（直接复用预计算数据）
%  ===============================================================
total_tic   = tic;
combo_count = 0;
total_combos = 7 * 7 * 7 * 4;

for j1 = 1:7
    for j2 = 1:7
        for j3 = 1:7
            for j4 = 1:4
                alpha = list(j1);
                beta  = list(j2);
                gamma = list(j3);
                tao   = fix(list1(j4) * num_feature);

                results = zeros(nfold, 7);
                t_param = tic;

                for i = 1:nfold
                    %% ===== 调用优化后的 LRALB，传入 precomp =====
                    [H, ~] = LRALB(fold_train_data{i}, ...
                        fold_train_target_pmfs{i}, ...
                        fold_precomp{i}.R, ...   % 已预处理的R
                        fold_S_A{i}, ...
                        alpha, beta, gamma, tao, ...
                        fold_precomp{i});         % ← 第9个参数

                    % 特征选择
                    [~, index] = sort(sum(H .* H, 2), 'descend');
                    f = index(1:k_select);

                    % MLKNN 评估
                    [Prior, PriorN, Cond, CondN] = MLKNN_train(...
                        fold_train_data{i}(:, f), fold_train_target_ml{i}, 10, 1);
                    [HL, RL, Cov, AP, macf1, micf1, OE, ~, ~] = MLKNN_test(...
                        fold_train_data{i}(:, f), fold_train_target_ml{i}, ...
                        fold_test_data{i}(:, f), fold_test_target_ml{i}, ...
                        10, Prior, PriorN, Cond, CondN);

                    results(i, :) = [HL, RL, OE, AP, macf1, micf1, Cov];
                end

                time1 = toc(t_param);
                rr  = mean(results);
                rr2 = std(results);

                combo_count = combo_count + 1;
                if mod(combo_count, 50) == 0 || rr(4) > ACRmax
                    fprintf('[%d/%d] | α=%.3f β=%.3f γ=%.3f tao=%.1f | AP=%.4f | Best=%.4f | %.1fs\n', ...
                        combo_count, total_combos, ...
                        alpha, beta, gamma, list1(j4), rr(4), ACRmax, time1);
                end

                if rr(4) > ACRmax
                    ACRmax    = rr(4);
                    zyrr      = rr;
                    zyrr2     = rr2;
                    bestalpha = alpha;
                    bestbeta  = beta;
                    bestgamma = gamma;
                    besttao   = list1(j4);
                    best_time = time1;
                end
            end
        end
    end
end

total_elapsed = toc(total_tic);
fprintf('最优参数: α=%.3f, β=%.3f, γ=%.3f, tao=%.1f, AP=%.4f\n\n', ...
    bestalpha, bestbeta, bestgamma, besttao, ACRmax);

