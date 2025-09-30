# Jogo da Velha Dinâmico com IA Uma Investigação sobre Estratégia e Convergência

## Objetivo

Este projeto explora a resolução de uma variação dinâmica do Jogo da Velha, utilizando Aprendizagem por Reforço (Q-Learning). O objetivo principal foi conseguir resolver essa variante do jogo da velha e no caminho aprender sobre aprendizado de maquina

## Jogo da Velha Dinâmico

A diferença principal entre este estilo de jogo da velha e o original é:
* **Limite de 3 Peças:** Cada jogador ('X' e 'O') pode ter um máximo de 3 peças no tabuleiro em simultâneo.
* **Remoção Dinâmica:** Ao realizar a 4ª jogada (e as seguintes), a peça mais antiga do jogador é automaticamente removida do tabuleiro ao mesmo tempo que a nova é colocada.

## Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Bibliotecas Principais:**
    * `collections.deque`: Para controlar a regra de remoção da peça mais antiga.
    * `pickle`: Para armazenar e carregar o "cérebro" treinado da IA.

## Metodologia

O projeto foi fundamentado em 2 classes essenciais, o `Board` e o `Agent`.

O `Board` é resposavel por toda a logica do jogo, ele possui o gerenciamento de jogadas, ordem dos jogadores, e o estado atual. O estado é composto por 3 elementos:

1.  A configuração visual do tabuleiro (uma tupla de 9 posições).
2.  A ordem das peças do jogador 'X' (uma tupla de tuplas de coordenadas).
3.  A ordem das peças do jogador 'O' (uma tupla de tuplas de coordenadas).

### Aprendizagem por Reforço com Q-Learning

A classe `Agent` é responsavel por armazenar o "cerebro" da IA, ele utiliza o algoritmo **Q-Learning**, uma técnica de Aprendizagem por Reforço que não exige um modelo prévio do ambiente (`model-free`).

#### A Q-Table (O "Cérebro" da IA)

O conhecimento do agente é armazenado em uma estrutura de dados chamada **Q-Table**, implementada como um dicionário de dicionários em Python (`q_table[estado][ação] = valor_q`).

* **estado**: A tupla completa que descreve o estado do jogo.
* **ação**: Uma jogada específica, representada por uma tupla de coordenadas `(linha, coluna)`.
* **valor_q**: Um número que representa a "qualidade" ou a "recompensa futura esperada" de se tomar aquela `ação` naquele `estado`.

A atualização dos valores na Q-Table é feita através da fórmula do Q-Learning:

`novo_valor = valor_antigo + taxa_aprendizado * (recompensa_futura - valor_antigo)`

Esta fórmula é a forma que IA realmente aprende. No final de cada partida, a IA recapitula o seu histórico de jogadas e aplica esta fórmula a cada uma delas, de trás para a frente. Vamos desconstruí-la:

`(recompensa_futura - valor_antigo)`: Esta parte calcula o "erro de previsão" da IA. É a diferença entre a recompensa que ela realmente obteve no futuro e o valor que ela achava que ia obter.

`taxa_aprendizado * ( ... )`: A IA não corrige o seu conhecimento de uma só vez. A taxa_aprendizado indica o quao sucetivel ela esta a mudar a sua forma de pensar, um valor mais baixo, garante que é necessario validar o conhecimento, garantindo que a IA faça apenas um pequeno ajuste na direção certa. Isto torna o aprendizado mais estável e consistente.

`valor_antigo + ...`: Aqui o novo valor é simplesmente o valor antigo mais este pequeno ajuste. É assim que, jogo após jogo, a IA vai refinando as suas opiniões sobre cada jogada até que elas direcionem para a estratégia ótima.

#### Parâmetros de Treino

* **Taxa de Aprendizado (`learning_rate`): 0.1** - Garante um aprendizado estável e consistente, evitando que uma única partida estranha corrompesse o conhecimento já adquirido.
* **Fator de Desconto (`discount_factor`): 0.9** - Incentiva a IA a pensar a longo prazo, valorizando recompensas futuras.
* **Nível de Exploração (Epsilon-Decay):** Uma estratégia de decaimento linear foi implementada, começando em 100% de exploração e diminuindo gradualmente até um mínimo de 1%, garantindo que a IA nunca pare de explorar completamente.
* **Sistema de Recompensas:** Vitória: +1, Derrota: -1, Empate: 0.

---

A investigação foi uma jornada de otimização, revelando que a qualidade da metodologia de treino é exponencialmente mais importante do que a *quantidade* bruta de simulações.

### A falha da metodologia antiga

Os primeiros modelos de IA foram treinados com uma curva de aprendizado (decaimento do Epsilon) padrão. Os resultados mostraram que, mesmo com centenas de milhões de partidas, a IA ficava "presa" em estratégias sub-ótimas. Ela aprendia a jogar bem, mas não de forma perfeita, e demonstrava fragilidades contra jogadas completamente aleatorias. A conclusão inicial foi que resolver o jogo era extremamente complexo e exigia um volume de treino massivo para ser descoberta.

### Encontrando a sequencia perfeita

Após aproximadamente 750 Milhões de testes, a IA conseguiu encontrar a soluçao perfeita, mas o mais surpreendente é que a solução apesar de não tão obvia, era relativamente curta, o que levantou a duvida do por que ter demorando tanto tempo para ser encontrada, com isso apareceu a duvida se era possivel otmizar o treinamento da IA.

## A otimização

A principal falha que a IA estava cometendo era ficar preso em um estilo de jogo que, apesar de ser muito consistente, não era a solução perfeita. A solução encontrada foi forçar ela a passar uma parte maior do seu treino com um nivel de exploração mais alto, permitindo que ela encontrasse varios caminhos otimos, precisando apenas refinar eles no final do treino, assim descobrindo a solução perfeita.

O resultado desta otimização foi uma descoberta surpreendente:

**Uma IA treinada com o novo decaimento por apenas 100 mil conseguiu descobrir a mesma estratégia perfeita, que a antiga IA precisou de mais de 750 milhões de partidas. Isto representa uma melhoria na eficiência do treino de mais de 5000x.**

Esta descoberta prova que o fator limitante não era a complexidade do jogo, mas sim a eficiência do método de aprendizado.

### A Sequência Invencível

A análise das partidas vitoriosas da IA otimizada revelou que existem apenas 5 sequencias possiveis de jogadas em que todas levam a vitória do primeiro jogador ('X') com uma sequencia de jogadas forçadas a partir da segunda jogada.

**A Abertura Perfeita:**
A análise mostrou que a IA 'X' inicia o jogo com a jogada **(1, 2)** (ou as suas simétricas) em **100%** das vezes.

**A Sequência da Vitória:**
<details>
<summary><strong>Ver Cenário 1: Resposta no Meio (1, 1)</strong></summary>

<pre><code>
Jogada 1 (IA 'X'): joga em (1, 2)
A IA estabelece a sua primeira peça. X1 é a sua peça mais antiga.
-------------
|   |   |   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 2 (Você 'O'): joga em (1, 1)
Uma jogada lógica, a ocupar o centro.
-------------
|   |   |   |
-------------
|   | O1| X1|
-------------
|   |   |   |
-------------
Jogada 3 (IA 'X'): joga em (0, 2)
A IA cria uma ameaça imediata na coluna da direita.
-------------
|   |   | X2|
-------------
|   | O1| X1|
-------------
|   |   |   |
-------------
Jogada 4 (Você 'O'): joga em (2, 2)
Você é forçado a bloquear a coluna da direita para não perder.
-------------
|   |   | X2|
-------------
|   | O1| X1|
-------------
|   |   | O2|
-------------
Jogada 5 (IA 'X'): joga em (0, 0)
A IA estabelece a sua terceira peça, criando uma nova ameaça na diagonal.
-------------
| X3|   | X2|
-------------
|   | O1| X1|
-------------
|   |   | O2|
-------------
Jogada 6 (Você 'O'): joga em (0, 1)
Novamente, uma jogada forçada. Você tem de bloquear a linha de cima.
-------------
| X3| O3| X2|
-------------
|   | O1| X1|
-------------
|   |   | O2|
-------------
Jogada 7 (IA 'X'): joga em (2, 0)
A peça mais antiga de X, a X1 em (1,2), desaparece. A IA cria uma nova ameaça na primeira coluna.
-------------
| X2| O3| X1|
-------------
|   | O1|   |
-------------
| X3|   | O2|
-------------
Jogada 8 (Você 'O'): joga em (1, 0)
A sua peça mais antiga, O1 em (1,1), some. Você é forçado a bloquear a coluna da esquerda.
-------------
| X2| O2| X1|
-------------
| O3|   |   |
-------------
| X3|   | O1|
-------------
Jogada 9 (IA 'X'): joga em (2, 1)
A peça mais antiga de X, X1 em (0,2), some. A IA agora cria uma ameaça linha de baixo. É impossível defender
-------------
| X1| O2|   |
-------------
| O3|   |   |
-------------
| X2| X3| O1|
-------------
Jogada 10 (Você 'O'): joga em (0, 2)
A sua peça O1 em (2,2) some. E a armadilha já está montada.
-------------
|   | O1| O3|
-------------
| O2|   |   |
-------------
| X2| X3|   |
-------------
Jogada 11 (IA 'X'): joga em (2, 2)
A peça X1 em (0,0) some. A IA coloca a peça final para a vitória.
-------------
|   | O1| O3|
-------------
| O2|   |   |
-------------
| X1| X2| X3| 
-------------

</code></pre>
</details>

<details>
<summary><strong>Ver Cenário 2: Resposta no Canto (0, 0)</strong></summary>

<pre><code>
Jogada 1 (X): joga em (1, 2)
-------------
|   |   |   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 2 (O): joga em (0, 0)
-------------
| O1|   |   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 3 (X): joga em (1, 1)
-------------
| O1|   |   |
-------------
|   | X2| X1|
-------------
|   |   |   |
-------------
Jogada 4 (O): joga em (1, 0)
-------------
| O1|   |   |
-------------
| O2| X2| X1|
-------------
|   |   |   |
-------------
Jogada 5 (X): joga em (2, 0)
-------------
| O1|   |   |
-------------
| O2| X2| X1|
-------------
| X3|   |   |
-------------
Jogada 6 (O): joga em (0, 2)
-------------
| O1|   | O3|
-------------
| O2| X2| X1|
-------------
| X3|   |   |
-------------
Jogada 7 (X): joga em (2, 2)
(A peça mais antiga de X, a X1 em (1,2), desaparece)
-------------
| O1|   | O3|
-------------
| O2| X2|   |
-------------
| X3|   | X4|
-------------
Jogada 8 (O): joga em (2, 1)
(A peça mais antiga de O, a O1 em (0,0), desaparece)
-------------
|   |   | O3|
-------------
| O2| X2|   |
-------------
| X3| O4| X4|
-------------
Jogada 9 (X): joga em (0, 0)
(A peça mais antiga de X, a X2 em (1,1), desaparece)
-------------
| X5|   | O3|
-------------
| O2|   |   |
-------------
| X3| O4| X4|
-------------
Jogada 10 (O): joga em (1, 1)
(A peça mais antiga de O, a O2 em (1,0), desaparece)
-------------
| X5|   | O3|
-------------
|   | O5|   |
-------------
| X3| O4| X4|
-------------
Jogada 11 (X): joga em (0, 1)
(A peça mais antiga de X, a X3 em (2,0), desaparece)
-------------
| X5| X6| O3|
-------------
|   | O5|   |
-------------
|   | O4| X4|
-------------
Jogada 12 (O): joga em (2, 0)
(A peça mais antiga de O, a O3 em (0,2), desaparece)
-------------
| X5| X6|   |
-------------
|   | O5|   |
-------------
| O6| O4| X4|
-------------
Jogada 13 (X): joga em (0, 2)
(A peça mais antiga de X, a X4 em (2,2), desaparece)
-------------
| X5| X6| X7|
-------------
|   | O5|   |
-------------
| O6| O4|   |
-------------

</code></pre>
</details>

<details>
<summary><strong>Ver Cenário 3: Resposta no Meio-Cima (0, 1)</strong></summary>
<pre><code>
Jogada 1 (IA 'X'): joga em (1, 2)
A IA estabelece a sua primeira peça. X1 é a sua peça mais antiga.
-------------
|   |   |   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 2 (Você 'O'): joga em (0, 1)
Você ocupa uma posição central na linha de cima.
-------------
|   | O1|   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 3 (IA 'X'): joga em (0, 2)
A IA cria uma ameaça imediata na coluna da direita.
-------------
|   | O1| X2|
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 4 (Você 'O'): joga em (2, 2)
Você é forçado a bloquear a coluna da direita para não perder.
-------------
|   | O1| X2|
-------------
|   |   | X1|
-------------
|   |   | O2|
-------------
Jogada 5 (IA 'X'): joga em (1, 1)
A IA ocupa o centro, criando uma nova ameaça na diagonal.
-------------
|   | O1| X2|
-------------
|   | X3| X1|
-------------
|   |   | O2|
-------------
Jogada 6 (Você 'O'): joga em (2, 0)
Novamente, uma jogada forçada. Você tem de bloquear a ameaça diagonal criada pela IA.
-------------
|   | O1| X2|
-------------
|   | X3| X1|
-------------
| O3|   | O2|
-------------
Jogada 7 (IA 'X'): joga em (2, 1)
A peça mais antiga de X, a X1 em (1,2), desaparece. A IA usa esta jogada para começar a construir uma base na linha de baixo.
-------------
|   | O1| X1|
-------------
|   | X2|   |
-------------
| O3| X3| O2|
-------------
Jogada 8 (Você 'O'): joga em (1, 0)
A sua peça mais antiga, O1 em (0,1), some. obrigando você a abrir o espaço
-------------
|   |   | X1|
-------------
| O3| X2|   |
-------------
| O2| X3| O1|
-------------
Jogada 9 (IA 'X'): joga em (0, 1)
A peça mais antiga de X, a X1 em (0,2), some. E a IA vence
-------------
|   | X3|   |
-------------
| O3| X1|   |
-------------
| O2| X2| O1|
-------------

</code></pre>
</details>

<details>
<summary><strong>Ver Cenário 4: Resposta Adjacente (0, 2)</strong></summary>

<pre><code>
Jogada 1 (IA 'X'): joga em (1, 2)
-------------
|   |   |   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 2 (Você 'O'): joga em (0, 2)
-------------
|   |   | O1|
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 3 (IA 'X'): joga em (1, 1)
A IA ignora a sua ameaça e ocupa o centro.
-------------
|   |   | O1|
-------------
|   | X2| X1|
-------------
|   |   |   |
-------------
Jogada 4 (Você 'O'): joga em (1, 0)
Uma jogada forçada. Você tem de bloquear a linha do meio.
-------------
|   |   | O1|
-------------
| O2| X2| X1|
-------------
|   |   |   |
-------------
Jogada 5 (IA 'X'): joga em (0, 1)
A IA completa a sua estrutura de ataque inicial.
-------------
|   | X3| O1|
-------------
| O2| X2| X1|
-------------
|   |   |   |
-------------
Jogada 6 (Você 'O'): joga em (2, 1)
Outra jogada forçada para bloquear a coluna central.
-------------
|   | X3| O1|
-------------
| O2| X2| X1|
-------------
|   | O3|   |
-------------
Jogada 7 (IA 'X'): joga em (0, 0)
A peça mais antiga de X, a X1 em (1,2), desaparece.
-------------
| X3| X2| O1|
-------------
| O2| X1|   |
-------------
|   | O3|   |
-------------
Jogada 8 (Você 'O'): joga em (2, 2)
A sua peça mais antiga, O1 em (0,2), some. Você é forçado a defender-se.
-------------
| X3| X2|   |
-------------
| O1| X1|   |
-------------
|   | O2| O3|
-------------
Jogada 9 (IA 'X'): joga em (0, 2)
A peça mais antiga de X, a X1 em (1,1), some.
-------------
| X2| X1| X3|
-------------
| O1|   |   |
-------------
|   | O2| O3|
-------------

</code></pre>
</details>

<details>
<summary><strong>Ver Cenário 5: Resposta Oposta (1, 0)</strong></summary>
<pre><code>
Jogada 1 (IA 'X'): joga em (1, 2)
A IA estabelece a sua primeira peça na lateral.
-------------
|   |   |   |
-------------
|   |   | X1|
-------------
|   |   |   |
-------------
Jogada 2 (Você 'O'): joga em (1, 0)
-------------
|   |   |   |
-------------
| O1|   | X1|
-------------
|   |   |   |
-------------
Jogada 3 (IA 'X'): joga em (0, 2)
A IA cria uma ameaça imediata na coluna da direita.
-------------
|   |   | X2|
-------------
| O1|   | X1|
-------------
|   |   |   |
-------------
Jogada 4 (Você 'O'): joga em (2, 2)
Você é forçado a bloquear a coluna da direita.
-------------
|   |   | X2|
-------------
| O1|   | X1|
-------------
|   |   | O2|
-------------
Jogada 5 (IA 'X'): joga em (0, 0)
A IA estabelece a sua terceira peça, a controlar os cantos.
-------------
| X3|   | X2|
-------------
| O1|   | X1|
-------------
|   |   | O2|
-------------
Jogada 6 (Você 'O'): joga em (0, 1)
Novamente, uma jogada forçada para bloquear a linha de cima.
-------------
| X3| O3| X2|
-------------
| O1|   | X1|
-------------
|   |   | O2|
-------------
Jogada 7 (IA 'X'): joga em (2, 0)
A peça mais antiga de X, a X1 em (1,2), desaparece. A IA agora controla três dos quatro cantos.
-------------
| X2| O3| X1|
-------------
| O1|   |   |
-------------
| X3|   | O2|
-------------
Jogada 8 (Você 'O'): joga em (1, 1)
A sua peça mais antiga, O1 em (1,0), some. Você ocupa o centro.
-------------
| X2| O2| X1|
-------------
|   | O3|   |
-------------
| X3|   | O1|
-------------
Jogada 9 (IA 'X'): joga em (1, 0)
A peça mais antiga de X, a X1 em (0,2), some. E mais uma vitoria forçada.
-------------
| X1| O2|   |
-------------
| X3| O3|   |
-------------
| X2|   | O1|
-------------
</code></pre>
</details> 

## Conclusão Final

O "Jogo da Velha Dinâmico" foi **fortemente resolvido**. A investigação provou que existe uma **vantagem injusta para o primeiro jogador ('X')**, no qual permite que ele vença 100% das partidas caso conheça a sequencia perfeita.

O projeto não só produziu uma IA capaz de jogar de forma perfeita, como também serviu como uma ferramenta de investigação para descobrir as propriedades matemáticas e a solução de um jogo complexo a partir do zero.

## Como Usar este Projeto
O projeto está dividido em três arquivos principais:
1.  **`trainer.py`**: O motor de treino. Execute este ficheiro para treinar as IAs especialistas do zero.
2.  **`main.py`**: A interface gráfica do jogo. Execute este ficheiro para jogar contra a IA de 1B de treinos.
3.  **`tournament.py`**: Ferramenta de análise para executar batalhas entre as IAs.
