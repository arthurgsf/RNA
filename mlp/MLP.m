classdef MLP
    properties
        WHI
        BIAS_HI
        WOH
        BIAS_OH
        LEARNING_RATE = 0.01
        TRAIN_ERROR = [0;0;0;0]
        VALID_ERROR = [0;0;0;0]
        ACTIVATION
        FDFH
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

        function model = UpdateWeights(model, X, y)
            X = X';
            % camada escondida
            net_h = model.WHI * X + model.BIAS_HI;
            Yh = model.ACTIVATION(net_h);

            % Camada de Saida
            net_o = model.WOH * Yh + model.BIAS_OH;
            Yo = 1 * net_o;
            
            %Erro de Saida
            Eo = y - Yo;
            model.TRAIN_ERROR = model.TRAIN_ERROR + Eo.^2;

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
            n = size(y, 1);
            y = y';
            for i = 1:n
                model = UpdateWeights(model, X(i, :), y(:, i));
            end
            model.TRAIN_ERROR = model.TRAIN_ERROR/n;
        end

        function model = Validate(model, X, y)
            n = size(y, 1);
            X = X';
            y = y';
            for i = 1:n
                %Camada Escondida
                net_h = model.WHI * X(:, i) + model.BIAS_HI;
                Yh = model.ACTIVATION(net_h);
        
                % Camda de Saida
                net_o = model.WOH * Yh + model.BIAS_OH;
                Eo = y(:, i) - net_o;
                model.VALID_ERROR = model.VALID_ERROR + Eo.^2;
            end
            model.VALID_ERROR = model.VALID_ERROR/n;
        end
    end
end