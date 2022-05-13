classdef RBF
    properties
        HistoricoEq = []
        CENTROS
        WOH
        BIAS_OH
        LEARNING_RATE = 0.1
        HistoricoErroTreino = []
        HistoricoErroValid = []
        ABERTURAS
    end

    methods
        function obj = RBF(InputShape, OutputShape, HiddenNeurons, LR)
            % inicializacao dos pesos
            % centro (INPUT_SHAPE x HIDDEN_NEURONS)
            obj.CENTROS = (-2).* rand([InputShape, HiddenNeurons]) + 1;
            obj.ABERTURAS = zeros([1, HiddenNeurons]);

            obj.WOH = rand([OutputShape, HiddenNeurons]) - 0.5;
            obj.BIAS_OH = rand([OutputShape, 1]) - 0.5;
            
            obj.LEARNING_RATE = LR;
        end

        function model = AtualizarCentros(model, X)
            n = size(X, 2);
            while true
                for i = 1:n
                    idx_winner = DescobrirGanhador(model, X(:, i));
                    %atualiza o ganhador
                    model = AtualizarGanhador(model, idx_winner, X(:, i));
                end
    
                % Repete atualizacao de centros ate estabilizar erro de qntz
                model.HistoricoEq = [model.HistoricoEq; CalcularErroQuantizacao(model, X)];
                if size(model.HistoricoEq, 1) > 9
                    desvio = std(model.HistoricoEq(end-5:end));
                    desvio = round(desvio, 4);
                    if desvio == 0
                        break;
                    end
                end
            end
        end

        function Eq = CalcularErroQuantizacao(model, X)
            n = size(X, 2);
            Eq = 0;
            for i = 1:n
                Eq = Eq + norm(X(:, i) - DescobrirGanhador(model, X(:, i)))^2;
            end
            Eq = Eq / n;
        end
        
        function model = AtualizarGanhador(model, idx, x)
            model.CENTROS(:, idx) = model.CENTROS(:, idx) + model.LEARNING_RATE * (x - model.CENTROS(:, idx));
        end

        function idx_winner = DescobrirGanhador(model, x)
            %calcular centro mais proximo de x
            n = size(model.CENTROS, 2);
            idx_winner = 1;
            distance_winner = norm(model.CENTROS(:, 1) - x);

            for i = 2:n
                distance_compare = norm(model.CENTROS(:, i) - x);
                if distance_compare <= distance_winner
                    idx_winner = i;
                    distance_winner = distance_compare;
                end
            end
        end

        function model = CalcularAberturas(model)
            % metade da distância ate o centro mais perto
            n = size(model.CENTROS, 2);

            for i = 1:n

                % calcula a distancia minima
                dmin = -1;
                for j = 1:n
                    if j ~= i
                        dtemp = norm(model.CENTROS(:, i) - model.CENTROS(:, j));
                        if dmin == -1 || dtemp < dmin
                            dmin = dtemp;
                        end
                    end
                end
                model.ABERTURAS(:, i) = dmin/2;
            end
        end

        function model = AtualizarPesos(model, x, y)
            % Calcula Saida da Camada Escondida

            %primeiro calcula todos os expoentes
            n_centros = size(model.CENTROS, 2);
            distancias = zeros([1, n_centros]);

            for i = 1:n_centros
                distancias(1, i) = norm(x - model.CENTROS(:, i));
            end

            potencias = (distancias.^2)./(model.ABERTURAS);
            
            % Yh e um array com o resultado de 1/e elevado as potencias
            Yh = arrayfun(@(x) 1/exp(1)^(x), potencias);

            % Camada de Saida
            net_o = model.WOH * Yh' + model.BIAS_OH;
            Yo = 1 * net_o;
            
            %Erro de Saida
            Eo = y - Yo;
            model.HistoricoErroTreino = [model.HistoricoErroTreino, mean(Eo.^2)];

            % Variação dos pesos entre Out e Hidden
            DeltaWoh = model.LEARNING_RATE *(Eo * 1) * Yh;
            DeltaBiasOh = model.LEARNING_RATE * sum(Eo);

            % Atualiza os pesos
            model.WOH = model.WOH + DeltaWoh;
            model.BIAS_OH = model.BIAS_OH + DeltaBiasOh;
        end

        function model = Fit(model, X, y)
            n = size(X, 1);
            for i = 1:n
                X(i) = normalize(X(i), 'range', [0, 1]); %default [0 - 1]
            end
            model = AtualizarCentros(model, X);
            model = CalcularAberturas(model);

            n = size(X, 2);
            for i = 1:n
                model = AtualizarPesos(model, X(:, i), y(:, i));
            end
        end

        function model = Validate(model, X, y)
            n = size(X, 2);

            for i = 1:n
                x = X(:, i);
                % Calcula Saida da Camada Escondida

                %primeiro calcula todos os expoentes
                n_centros = size(model.CENTROS, 2);
                distancias = zeros([1, n_centros]);
                
                for j = 1:n_centros
                    distancias(1, j) = norm(x - model.CENTROS(:, j));
                end
    
                potencias = (distancias.^2)./(model.ABERTURAS);
                
                % Yh e um array com o resultado de 1/e elevado as potencias
                Yh = arrayfun(@(x) 1/exp(1)^(x), potencias);
        
                % Camda de Saida
                net_o = model.WOH * Yh' + model.BIAS_OH;
                Eo = y(:, i) - net_o;
                model.HistoricoErroValid = [model.HistoricoErroValid, mean(Eo.^2)];
            end
        end
    end
end