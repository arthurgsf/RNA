%{
    Aquisicao dos dados
%}

% leitura do arquivo csv
opts = detectImportOptions("car.data", "FileType","delimitedtext");
opts = setvartype(opts, "Var3", 'string');  %or 'char' if you prefer
opts = setvartype(opts, "Var4", 'string');  %or 'char' if you prefer

X = readtable("car.data", opts);

% separando as labels dos atributos
y = X.Var7;
X = removevars(X, "Var7");

% pre-processamento das labels (transformando em numeros [0-3])
y = cellfun(@(x) pre_Y(x), y, "UniformOutput",false);

% pre-processamento das variaveis
% buying
X.Var1 = cellfun(@(v) pre_buying_maint(v), X.Var1);
%maint
X.Var2 = cellfun(@(v) pre_buying_maint(v), X.Var2);
%doors
X.Var3 = cellfun(@(v) pre_doors(v), X.Var3);
%persons
X.Var4 = cellfun(@(v) pre_persons(v), X.Var4);
%lug boot
X.Var5 = cellfun(@(v) pre_lug_boot(v), X.Var5);
%safety
X.Var6 = cellfun(@(v) pre_safety(v), X.Var6);

X = table2array(X);
y = cell2mat(y);

%{
    Treinamento da rede
%}
INPUT_SHAPE = 6;
OUTPUT_SHAPE = 4;
HIDDEN_NEURONS = 5;
EPOCHS = 100;

HistoricoErroTreino = [];
HistoricoErroValid = [];
HistoricoMediaTreino = [];
HistoricoMediaValid = [];



for i = 1:EPOCHS
    [X_train, y_train, X_valid, y_valid] = train_test_split(X, y, 0.2);
    model = RBF(INPUT_SHAPE, OUTPUT_SHAPE, HIDDEN_NEURONS, 0.01);
    model = Fit(model, X_train', y_train');
    model = Validate(model, X_valid', y_valid');
    
    HistoricoErroTreino = [HistoricoErroTreino, model.HistoricoErroTreino];
    HistoricoErroValid = [HistoricoErroValid, model.HistoricoErroValid];
    if mod(i, 10) == 0
        HIDDEN_NEURONS = HIDDEN_NEURONS + 1;
        HistoricoMediaTreino = [HistoricoMediaTreino, mean(HistoricoErroTreino(i - 9: i))];
        HistoricoMediaValid = [HistoricoMediaValid, mean(HistoricoErroValid(i - 9: i))];
    end
end

HistoricoMediaTreino
HistoricoMediaValid

figure
hold on
plot([1:10], HistoricoMediaTreino, 'r', [1:10], HistoricoMediaValid, 'b');
legend('treino', 'validação');
title("Erro médio ao longo de 100 Treinos");

%{
    Funcs de pre-processamento
%}
function [X_train, y_train, X_valid, y_valid] = train_test_split(X, y, rate)
    n = size(X, 1);
    idx = 1:n;
    train_idx = randi([1, n], [1, floor(n *(1- rate))]);
    valid_idx = setdiff(idx, train_idx);

    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
    X_valid = X(valid_idx, :);
    y_valid = y(valid_idx, :);
end


function V = pre_Y(v)
    if strcmp(v, "unacc") == 1
        V = [0; 0; 0; 1];
    elseif strcmp(v, "acc") == 1
        V = [0; 0; 1; 0];
    elseif strcmp(v, "good") == 1
        V = [0; 1; 0; 0];
    elseif strcmp(v, "vgood") == 1
        V = [1; 0; 0; 0];
    end
end

function V = pre_buying_maint(v)
    if strcmp(v, "vhigh") == 1
        V = 0;
    elseif strcmp(v, "high") == 1
        V = 1;
    elseif strcmp(v, "med") == 1
        V = 2;
    elseif strcmp(v, "low") == 1
        V = 3;
    else
        V = -1;
        class(v)
    end
end

function V = pre_doors(v)
    if strcmp(v, "2") == 1
        V = 0;
    elseif strcmp(v, "3") == 1
        V = 1;
    elseif strcmp(v, "4") == 1
        V = 2;
    elseif strcmp(v,  "5more") == 1
        V = 3;
    end
end

function V = pre_persons(v)
    if strcmp(v,  "2") == 1
        V = 0;
    elseif strcmp(v,  "4") == 1
        V = 1;
    elseif strcmp(v,  "more") == 1
        V = 2;
    end
end

function V = pre_lug_boot(v)
    if strcmp(v,  "small") == 1
        V = 0;
    elseif strcmp(v,  "med") == 1
        V = 1;
    elseif strcmp(v,  "big") == 1
        V = 2;
    end
end

function V = pre_safety(v)
    if strcmp(v,  "low") == 1
        V = 0;
    elseif strcmp(v,  "med") == 1
        V = 1;
    elseif strcmp(v,  "high") == 1
        V = 2;
    end
end
