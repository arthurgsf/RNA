classdef MLP
    properties
        WHI
        WOH

        BIAS_HI
        BIAS_OH

        ACTIVATION
        FDFH

        LEARNING_RATE
        TRAIN_ERROR_HISTORY = {}
        VALIDATION_ERROR_HISTORY = {}
    end
    methods
        function obj = MLP(InputShape, OutputShape, HiddenNeurons, Activation, Fdfh, LR)
            % inicializacao dos pesos
            obj.WHI = rand([HiddenNeurons, InputShape]) - 0.5;
            obj.BIAS_HI = rand([HiddenNeurons, 1]) - 0.5;
            obj.WOH = rand([OutputShape, HiddenNeurons]) - 0.5;
            obj.BIAS_OH = rand([OutputShape, 1]) - 0.5;
            obj.ACTIVATION = Activation;
            obj.FDFH = Fdfh;
            obj.LEARNING_RATE = LR;
        end

        function [model, Eo] = UpdateWeights(model, X, y)
            % camada escondida
            net_h = model.WHI * X + model.BIAS_HI;
            Yh = model.ACTIVATION(net_h);

            % Camada de Saida
            net_o = model.WOH * Yh + model.BIAS_OH;
            Yo = 1 * net_o;
            
            %Erro de Saida
            Eo = y - Yo;
            
            % Variação dos pesos entre Out e Hidden
            DeltaWoh = model.LEARNING_RATE *(Eo * 1) * Yh';
            DeltaBiasOh = model.LEARNING_RATE * sum(Eo);

            % Backpropagation (Estimando Erro da Camada Escondida)
            Eh = -model.WOH' * (Eo * 1);
            
            % Variação dos pesos entre Hidden e Input
            dfh = model.FDFH(net_h);
            DeltaWhi = -model.LEARNING_RATE * (Eh .* dfh) * X';
            DeltaBiasHi = -model.LEARNING_RATE * sum(Eh .* dfh);

            % Atualiza os pesos
            model.WOH = model.WOH + DeltaWoh;
            model.WHI = model.WHI + DeltaWhi;
            model.BIAS_HI = model.BIAS_HI + DeltaBiasHi;
            model.BIAS_OH = model.BIAS_OH + DeltaBiasOh;
        end

        function model = Fit(model, X, y)
            n = size(y, 2);
            errors =  zeros([n, 1]);
            for i = 1:n
                [model, error] = UpdateWeights(model, X(:, i), y(:, i));
                errors(i) = sum(error.^2);
            end
            model.TRAIN_ERROR_HISTORY = [model.TRAIN_ERROR_HISTORY; mean(errors)];
        end

        function model = Validate(model, X, y)
            n = size(y, 2);
            errors =  zeros([n, 1]);
            for i = 1:n
                %Camada Escondida
                net_h = model.WHI * X(:, i) + model.BIAS_HI;
                Yh = model.ACTIVATION(net_h);
        
                % Camda de Saida
                net_o = model.WOH * Yh + model.BIAS_OH;
                Eo = y(:, i) - net_o;
                errors(i) = sum(Eo.^2);
                
            end
            model.VALIDATION_ERROR_HISTORY = [model.VALIDATION_ERROR_HISTORY ; mean(errors)];
        end

        function predictions = Predict(model, X)
            n = size(X, 2);
            predictions = zeros(size(model.WOH, 1), n);
            for i = 1:n
                %Camada Escondida
                net_h = model.WHI * X(:, i) + model.BIAS_HI;
                Yh = model.ACTIVATION(net_h);
        
                % Camda de Saida
                predictions(:, i) = model.WOH * Yh + model.BIAS_OH;
            end
        end
    end
end