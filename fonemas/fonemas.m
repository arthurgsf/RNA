[X, y] = load_data();

X = get_features(X);
y = get_labels(y);

[X_train, y_train, X_test, y_test] = train_test_split(X, y, 0.6);
[X_valid, y_valid, X_test, y_test] = train_test_split(X_test, y_test, 0.4);

EPOCHS = 1500;
HIDDEN_NEURONS = 15;
OUTPUT_SHAPE = 6;
INPUT_SHAPE = size(X, 1);

ACTIVATION = @(x) logsig(x); % Ativacao
FDFH = @(x) logsig(x)-(logsig(x).^2); % Derivada da Fn de Ativacao

model = MLP(INPUT_SHAPE, OUTPUT_SHAPE, HIDDEN_NEURONS, ACTIVATION, FDFH, 0.001);

for i = 1:EPOCHS
    model = Fit(model, X_train, y_train);
    model = Validate(model, X_valid, y_valid);
end

% do final test
y_pred = Predict(model, X_test);
y_test = reverse_labels(y_test);
y_pred = reverse_labels(y_pred);
cmat = confusionmat(y_test, y_pred);


figure
title('Erro de Treinamento e Validação')
hold on
plot(1:size(model.TRAIN_ERROR_HISTORY,1), cell2mat(model.TRAIN_ERROR_HISTORY), 'r');
plot(1:size(model.VALIDATION_ERROR_HISTORY,1), cell2mat(model.VALIDATION_ERROR_HISTORY), 'b');
legend('treino', 'validação');
xlabel("Épocas");
ylabel("MSE");


hold off
saveas(gcf, 'training_error.png')

figure
title('Matriz de Confusão')
confusionchart(cmat, ["di"; "rei"; "ta"; "es"; "quer"; "da"]);
saveas(gcf, 'matriz_confusao.png')


T_MIN_ERROR = min(cell2mat(model.TRAIN_ERROR_HISTORY))
V_MIN_ERROR = min(cell2mat(model.VALIDATION_ERROR_HISTORY))



function [X, y] = load_data()
    X = {};
    y = {};
    classes = ["di"; "rei"; "ta"; "es"; "quer"; "da"];
    for i = 1:size(classes)
        path = "./dataset/" + classes(i);
        files = dir(path);
        for j = 1:size(files)
            if files(j).isdir == false && contains(files(j).name, '.wav')
                audio = audioread(path + "/" + files(j).name);
                X = [X; audio(:, 1)];
                y = [y, classes(i)];
            end
        end
    end
end

function [X_train, y_train, X_valid, y_valid] = train_test_split(X, y, train_perc)
    n = size(X,2);
    p = randperm(n);

    train_size = ceil(train_perc * n);
    train_idx = p(1:train_size);
    valid_idx = p(train_size + 1:n);

    X_train = X(:, train_idx);
    y_train = y(:, train_idx);
    X_valid = X(:, valid_idx);
    y_valid = y(:, valid_idx);
end

function features = get_features(X)
    n = size(X, 1);
    temp_features = {};
    for i = 1:n
        t = abs(fft(cell2mat(X(i))));
        half = round(size(t, 1)/2);
        temp_features = [temp_features, t(1:half)];
    end
    bigger = max(cellfun(@length, temp_features));
    temp_features = cellfun(@(x) [x; zeros(bigger - numel(x), 1)], temp_features, 'un', 0);
    temp_features = cell2mat(temp_features);
    
    features = zeros([bigger/100, size(temp_features, 2)], "double");
    start = 1;
    final = 100;
    for i = 1: round(bigger/100)
        for j = 1: size(temp_features, 2)
            features(i, j) = mean(temp_features(start + 100* (i -1):final*i, j));
        end
    end
end

function labels = get_labels(y)
    m = containers.Map(["di", "rei", "ta", "es", "quer", "da"], [1, 2, 3, 4, 5, 6]);
    labels = zeros([6, size(y, 2)]);
    for i = 1:size(y,2)
        idx = m(y(1, i));
        labels(idx, i) = 1;
    end
end

function labels = reverse_labels(y)
    m = containers.Map([1, 2, 3, 4, 5, 6], ["di", "rei", "ta", "es", "quer", "da"]);
    labels = strings(size(y, 2), 1);
    for i = 1:size(y, 2)
        [~, idx] = max(y(:, i));
        labels(i) = m(idx);
    end
end