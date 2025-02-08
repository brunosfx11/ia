import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from io import StringIO

# --- Funções Auxiliares ---
#Função para obter os dados de multiplos arquivos
def obter_e_combinar_csvs(urls):
    """Obtém e combina vários CSVs a partir de uma lista de URLs."""
    all_dfs = []  # Lista para armazenar os DataFrames
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            try:
                content = response.content.decode('utf-8')
            except UnicodeDecodeError:
                content = response.content.decode('latin-1', errors='replace')

            df = pd.read_csv(StringIO(content))
            all_dfs.append(df)  # Adiciona o DataFrame à lista

        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar o CSV de {url}: {e}")
            #Nesse caso, não retorna None, pois outras URLs podem funcionar. Continua o loop.
        except pd.errors.ParserError as e:
            print(f"Erro ao analisar o CSV de {url}: {e}")
            #Mesmo tratamento do erro anterior.
        except Exception as e:
            print(f"Erro inesperado ao processar {url}: {e}")

    if all_dfs:  # Verifica se a lista não está vazia
        return pd.concat(all_dfs, ignore_index=True)  # Combina todos os DataFrames em um só
    else:
        print("Nenhum CSV válido foi baixado.")
        return None

def obter_csv_da_web(url):
    """Obtém um CSV a partir de uma URL e retorna um DataFrame do Pandas."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lança um erro para códigos de status HTTP ruins (4xx, 5xx)
        # Decodifica o conteúdo usando UTF-8 (com tratamento para erros)
        try:
            content = response.content.decode('utf-8')
        except UnicodeDecodeError:
            content = response.content.decode('latin-1', errors='replace')

        df = pd.read_csv(StringIO(content))  # StringIO do módulo io
        return df

    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o CSV: {e}")
        return None
    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o CSV: {e}")
        return None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None
# --- Funções para features ---

def calcular_medias_moveis(df, coluna_gols, janela):
    """Calcula as médias móveis de gols para cada time."""

    ultima_linha_time = {}
    medias_moveis = []
    num_jogos_time = {}

    for i, row in df.iterrows():
        time_casa = row['TimeCasa']
        time_visitante = row['TimeVisitante']

        # --- Time da Casa ---
        if time_casa not in ultima_linha_time:
            medias_moveis.append(np.nan)  # Primeiro jogo do time: NaN
            num_jogos_time[time_casa] = 0
        else:
            jogos_anteriores = df.loc[(df['TimeCasa'] == time_casa) | (df['TimeVisitante'] == time_casa)]
            jogos_anteriores = jogos_anteriores.loc[jogos_anteriores['Data'] < row['Data']] #Filtra por data
            if len(jogos_anteriores) >= janela:
                # Calcula a média dos últimos 'janela' jogos
                gols_anteriores = jogos_anteriores.tail(janela).apply(
                    lambda x: x['GolsCasa'] if x['TimeCasa'] == time_casa else x['GolsVisitante'], axis=1
                )
                media = gols_anteriores.mean()
                medias_moveis.append(media)
            else:
                medias_moveis.append(np.nan) #Não tem jogos suficientes
            num_jogos_time[time_casa] = len(jogos_anteriores)

        ultima_linha_time[time_casa] = i  # Atualiza a última linha do time da casa

        # --- Time Visitante --- (mesma lógica, mas para o time visitante)
        if time_visitante not in ultima_linha_time:
            medias_moveis.append(np.nan)
            num_jogos_time[time_visitante] = 0
        else:
            jogos_anteriores = df.loc[(df['TimeCasa'] == time_visitante) | (df['TimeVisitante'] == time_visitante)]
            jogos_anteriores = jogos_anteriores.loc[jogos_anteriores['Data'] < row['Data']] #Filtra por data.
            if len(jogos_anteriores) >= janela:
                gols_anteriores = jogos_anteriores.tail(janela).apply(
                    lambda x: x['GolsCasa'] if x['TimeCasa'] == time_visitante else x['GolsVisitante'], axis=1
                )
                media = gols_anteriores.mean()
                medias_moveis.append(media)
            else:
                medias_moveis.append(np.nan)
            num_jogos_time[time_visitante] = len(jogos_anteriores)
        ultima_linha_time[time_visitante] = i

    # Adiciona a coluna de média móvel ao DataFrame original
    df[f'MM_{coluna_gols}_{janela}'] = medias_moveis[::2]  # Pega um valor sim, outro não (pois duplicamos).
    return df, num_jogos_time

def calcular_desempenho_recente(df, janela, coluna_resultado='Resultado'):
    """Calcula vitórias/empates/derrotas recentes em casa/fora (ex: '2-1-2')."""

    resultados_casa = []
    resultados_fora = []

    for i, row in df.iterrows():
        time_casa = row['TimeCasa']
        time_visitante = row['TimeVisitante']

        # Jogos anteriores do time da casa *EM CASA*
        jogos_casa = df.loc[(df['TimeCasa'] == time_casa) & (df['Data'] < row['Data'])].tail(janela)
        # Jogos anteriores do time visitante *FORA*
        jogos_fora = df.loc[(df['TimeVisitante'] == time_visitante) & (df['Data'] < row['Data'])].tail(janela)

        # Contar vitórias, empates e derrotas em casa
        vitorias_casa = len(jogos_casa[jogos_casa[coluna_resultado] == 'H'])
        empates_casa = len(jogos_casa[jogos_casa[coluna_resultado] == 'D'])
        derrotas_casa = len(jogos_casa[jogos_casa[coluna_resultado] == 'A'])

        # Contar vitórias, empates e derrotas fora
        vitorias_fora = len(jogos_fora[jogos_fora[coluna_resultado] == 'A'])
        empates_fora = len(jogos_fora[jogos_fora[coluna_resultado] == 'D'])
        derrotas_fora = len(jogos_fora[jogos_fora[coluna_resultado] == 'H'])

        resultados_casa.append(f"{vitorias_casa}-{empates_casa}-{derrotas_casa}")
        resultados_fora.append(f"{vitorias_fora}-{empates_fora}-{derrotas_fora}")

    df['DesempenhoCasa'] = resultados_casa
    df['DesempenhoFora'] = resultados_fora
    return df
def calcular_confronto_direto(df, janela=5):
    """
    Calcula estatísticas de confronto direto entre os times para cada jogo.

    Args:
        df: DataFrame com os dados dos jogos.
        janela: Número de jogos anteriores a considerar no histórico.

    Returns:
        DataFrame com as colunas de confronto direto adicionadas.
    """

    historico_confrontos = {}  # Dicionário para guardar o histórico
    resultados_casa = []
    resultados_visitante = []


    for i, row in df.iterrows():  # Itera por cada linha (jogo) do DataFrame
        time_casa = row['TimeCasa']
        time_visitante = row['TimeVisitante']

        # A chave do histórico é uma tupla (time1, time2), ordenada alfabeticamente.
        # Isso garante que (Arsenal, Chelsea) e (Chelsea, Arsenal) usem a mesma chave.
        key = tuple(sorted((time_casa, time_visitante)))
         #Inicializa histórico:
        historico = []
        if key in historico_confrontos:
            # Obtém o histórico de confrontos, limitado à janela especificada.
            historico = historico_confrontos[key][-janela:]

        vitorias_casa = 0
        empates = 0
        derrotas_casa = 0
        for resultado in historico:
          # Ajusta a contagem com base em quem jogou em casa *NAQUELE* jogo.
          if resultado == 'H':
              if key[0] == time_casa:  # Time casa ganhou
                  vitorias_casa +=1
              else:                   # Time visitante ganhou
                  derrotas_casa += 1

          elif resultado == 'A':
              if key[0] == time_casa:  #Time casa perdeu
                  derrotas_casa += 1
              else:                     #Time visitante ganhou
                  vitorias_casa += 1
          else:
              empates += 1  # Empate não muda, independente de quem jogou em casa

        resultados_casa.append(f"{vitorias_casa}-{empates}-{derrotas_casa}")
        resultados_visitante.append(f"{len(historico)-vitorias_casa-empates}-{empates}-{vitorias_casa}")


        # Adiciona o resultado *deste* jogo ao histórico (para uso futuro).
        historico_confrontos.setdefault(key, []).append(row['Resultado']) #.setdefault() evita erros caso a chave não exista

    df['Confronto_Direto_Casa'] = resultados_casa
    df['Confronto_Direto_Visitante'] = resultados_visitante
    return df
def calcular_probabilidades_implicitas(df, colunas_odds):
  """Calcula as probabilidades implícitas a partir das odds (1 / odd)."""
  for col in colunas_odds:
      # Trata o caso de divisão por zero, atribuindo uma probabilidade pequena (0.0001)
      df[f'Prob_{col}'] = df[col].apply(lambda x: 1/x if (x != 0 and not pd.isna(x)) else 0.0001)

  # Exemplo: Diferença de probabilidade entre casa e visitante (usando Bet365)
  df['Prob_Diff'] = df['Prob_B365_Casa'] - df['Prob_B365_Visitante']
  return df
# --- Fase 1 e 2: Obtenção e Limpeza dos Dados ---

#Agora a função recebe uma lista de urls
urls_csv = ["https://www.football-data.co.uk/mmz4281/2324/E0.csv",
           "https://www.football-data.co.uk/mmz4281/2425/E0.csv"]

df = obter_e_combinar_csvs(urls_csv)


if df is not None:  # Continua somente se o DataFrame foi criado com sucesso
    colunas_map = {
        'Div': 'Divisao',
        'Date': 'Data',
        'Time': 'Hora',
        'HomeTeam': 'TimeCasa',
        'AwayTeam': 'TimeVisitante',
        'FTHG': 'GolsCasa',
        'FTAG': 'GolsVisitante',
        'FTR': 'Resultado',
        'HTHG': 'GolsCasaHT',
        'HTAG': 'GolsVisitanteHT',
        'HTR': 'ResultadoHT',
        'Referee': 'Arbitro',
        'HS': 'ChutesCasa',
        'AS': 'ChutesVisitante',
        'HST': 'ChutesGolCasa',
        'AST': 'ChutesGolVisitante',
        'HF': 'FaltasCasa',
        'AF': 'FaltasVisitante',
        'HC': 'EscanteiosCasa',
        'AC': 'EscanteiosVisitante',
        'HY': 'CartoesAmarelosCasa',
        'AY': 'CartoesAmarelosVisitante',
        'HR': 'CartoesVermelhosCasa',
        'AR': 'CartoesVermelhosVisitante',
        'B365H' : 'B365_Casa',
        'B365D' : 'B365_Empate',
        'B365A' : 'B365_Visitante',
        'PSH' :  'PS_Casa',  # Pinnacle
        'PSD' :  'PS_Empate',
        'PSA' :  'PS_Visitante',
        'AvgH' : 'BF_Casa', #BetFair
        'AvgD' : 'BF_Empate',
        'AvgA' : 'BF_Visitante',
    }

    df = df.rename(columns=colunas_map)
    colunas_manter = list(colunas_map.values())
    df = df[colunas_manter]
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['Data'], inplace=True)
    df['TotalGols'] = df['GolsCasa'] + df['GolsVisitante']

    colunas_odds = ['B365_Casa', 'B365_Empate', 'B365_Visitante',
                    'PS_Casa', 'PS_Empate', 'PS_Visitante',
                    'BF_Casa', 'BF_Empate', 'BF_Visitante']

    for col in colunas_odds:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in colunas_odds:
        df[col] = df[col].fillna(df[col].mean())


    # --- Fase 3: Análise Exploratória dos Dados --- (Opcional, mas útil para entender os dados)

    # Configurar o estilo dos gráficos (opcional)
    sns.set(style="whitegrid")

    # 1. Estatísticas Descritivas
    print("\nEstatísticas Descritivas:")
    print(df.describe())

    # 2. Distribuição de Resultados
    print("\nDistribuição de Resultados (H=Casa, D=Empate, A=Visitante):")
    print(df['Resultado'].value_counts())

    # --- Descomentar para gerar os gráficos
    #plt.figure(figsize=(6, 4))
    #sns.countplot(data=df, x='Resultado', order=['H', 'D', 'A'])
    #plt.title('Distribuição de Resultados dos Jogos')
    #plt.xlabel('Resultado')
    #plt.ylabel('Número de Jogos')
    #plt.show()

    # 3. Distribuição de Gols
    #plt.figure(figsize=(10, 6))
    #sns.histplot(data=df, x='GolsCasa', kde=True, binwidth=1)
    #plt.title('Distribuição de Gols do Time da Casa')
    #plt.xlabel('Gols')
    #plt.#plt.ylabel('Frequência')
    #plt.show()

    #plt.figure(figsize=(10, 6))
    #sns.histplot(data=df, x='GolsVisitante', kde=True, binwidth=1)
    #plt.title('Distribuição de Gols do Time Visitante')
    #plt.xlabel('Gols')
    #plt.ylabel('Frequência')
    #plt.show()

    #plt.figure(figsize=(10, 6))
    #sns.histplot(data=df, x='TotalGols', kde=True, binwidth=1)
    #plt.title('Distribuição do Total de Gols')
    #plt.xlabel('Total de Gols')
    #plt.ylabel('Frequência')
    #plt.show()

    # 4. Correlações
    #plt.figure(figsize=(12, 10))
    #sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    #plt.title('Matriz de Correlação')
    #plt.show()

    # --- Fase 4: Engenharia de Recursos ---

    # 1. Médias Móveis de Gols (Marcados e Sofridos)

    def calcular_medias_moveis(df, coluna_gols, janela):
        """Calcula as médias móveis de gols para cada time."""

        ultima_linha_time = {}
        medias_moveis = []
        num_jogos_time = {}

        for i, row in df.iterrows():
            time_casa = row['TimeCasa']
            time_visitante = row['TimeVisitante']

            # --- Time da Casa ---
            if time_casa not in ultima_linha_time:
                medias_moveis.append(np.nan)  # Primeiro jogo do time: NaN
                num_jogos_time[time_casa] = 0
            else:
                jogos_anteriores = df.loc[(df['TimeCasa'] == time_casa) | (df['TimeVisitante'] == time_casa)]
                jogos_anteriores = jogos_anteriores.loc[jogos_anteriores['Data'] < row['Data']] #Filtra por data
                if len(jogos_anteriores) >= janela:
                    # Calcula a média dos últimos 'janela' jogos
                    gols_anteriores = jogos_anteriores.tail(janela).apply(
                        lambda x: x['GolsCasa'] if x['TimeCasa'] == time_casa else x['GolsVisitante'], axis=1
                    )
                    media = gols_anteriores.mean()
                    medias_moveis.append(media)
                else:
                    medias_moveis.append(np.nan) #Não tem jogos suficientes
                num_jogos_time[time_casa] = len(jogos_anteriores)

            ultima_linha_time[time_casa] = i  # Atualiza a última linha do time da casa

            # --- Time Visitante --- (mesma lógica, mas para o time visitante)
            if time_visitante not in ultima_linha_time:
                medias_moveis.append(np.nan)
                num_jogos_time[time_visitante] = 0
            else:
                jogos_anteriores = df.loc[(df['TimeCasa'] == time_visitante) | (df['TimeVisitante'] == time_visitante)]
                jogos_anteriores = jogos_anteriores.loc[jogos_anteriores['Data'] < row['Data']] #Filtra por data.
                if len(jogos_anteriores) >= janela:
                    gols_anteriores = jogos_anteriores.tail(janela).apply(
                        lambda x: x['GolsCasa'] if x['TimeCasa'] == time_visitante else x['GolsVisitante'], axis=1
                    )
                    media = gols_anteriores.mean()
                    medias_moveis.append(media)
                else:
                    medias_moveis.append(np.nan)
                num_jogos_time[time_visitante] = len(jogos_anteriores)
            ultima_linha_time[time_visitante] = i

        # Adiciona a coluna de média móvel ao DataFrame original
        df[f'MM_{coluna_gols}_{janela}'] = medias_moveis[::2]  # Pega um valor sim, outro não (pois duplicamos).
        return df, num_jogos_time


    janela = 5
    df, _ = calcular_medias_moveis(df, 'GolsCasa', janela)
    df, _ = calcular_medias_moveis(df, 'GolsVisitante', janela)
    df['GolsSofridosCasa'] = df.groupby('TimeCasa')['GolsVisitante'].transform(pd.Series.shift)
    df['GolsSofridosVisitante'] = df.groupby('TimeVisitante')['GolsCasa'].transform(pd.Series.shift)
    df, _ = calcular_medias_moveis(df, 'GolsSofridosCasa', janela)
    df, num_jogos_sofridos = calcular_medias_moveis(df, 'GolsSofridosVisitante', janela)

    print("\nNúmero de jogos por time (Gols Marcados):")
    print(num_jogos_sofridos)

    df.drop(['GolsSofridosCasa', 'GolsSofridosVisitante'], axis=1, inplace=True)

    # --- Desempenho Recente em Casa/Fora ---

    def calcular_desempenho_recente(df, janela, coluna_resultado='Resultado'):
        """Calcula vitórias/empates/derrotas recentes em casa/fora (ex: '2-1-2')."""

        resultados_casa = []
        resultados_fora = []

        for i, row in df.iterrows():
            time_casa = row['TimeCasa']
            time_visitante = row['TimeVisitante']

            # Jogos anteriores do time da casa *EM CASA*
            jogos_casa = df.loc[(df['TimeCasa'] == time_casa) & (df['Data'] < row['Data'])].tail(janela)
            # Jogos anteriores do time visitante *FORA*
            jogos_fora = df.loc[(df['TimeVisitante'] == time_visitante) & (df['Data'] < row['Data'])].tail(janela)

            # Contar vitórias, empates e derrotas em casa
            vitorias_casa = len(jogos_casa[jogos_casa[coluna_resultado] == 'H'])
            empates_casa = len(jogos_casa[jogos_casa[coluna_resultado] == 'D'])
            derrotas_casa = len(jogos_casa[jogos_casa[coluna_resultado] == 'A'])

            # Contar vitórias, empates e derrotas fora
            vitorias_fora = len(jogos_fora[jogos_fora[coluna_resultado] == 'A'])
            empates_fora = len(jogos_fora[jogos_fora[coluna_resultado] == 'D'])
            derrotas_fora = len(jogos_fora[jogos_fora[coluna_resultado] == 'H'])

            resultados_casa.append(f"{vitorias_casa}-{empates_casa}-{derrotas_casa}")
            resultados_fora.append(f"{vitorias_fora}-{empates_fora}-{derrotas_fora}")

        df['DesempenhoCasa'] = resultados_casa
        df['DesempenhoFora'] = resultados_fora
        return df

    df = calcular_desempenho_recente(df, janela)
    #print(df[['Data', 'TimeCasa', 'TimeVisitante', 'Resultado', 'DesempenhoCasa', 'DesempenhoFora']].head(10))

    # --- Confronto Direto ---
    def calcular_confronto_direto(df, janela=5):
        """
        Calcula estatísticas de confronto direto entre os times para cada jogo.

        Args:
            df: DataFrame com os dados dos jogos.
            janela: Número de jogos anteriores a considerar no histórico.

        Returns:
            DataFrame com as colunas de confronto direto adicionadas.
        """

        historico_confrontos = {}  # Dicionário para guardar o histórico
        resultados_casa = []
        resultados_visitante = []


        for i, row in df.iterrows():  # Itera por cada linha (jogo) do DataFrame
            time_casa = row['TimeCasa']
            time_visitante = row['TimeVisitante']

            # A chave do histórico é uma tupla (time1, time2), ordenada alfabeticamente.
            # Isso garante que (Arsenal, Chelsea) e (Chelsea, Arsenal) usem a mesma chave.
            key = tuple(sorted((time_casa, time_visitante)))
             #Inicializa histórico:
            historico = []
            if key in historico_confrontos:
                # Obtém o histórico de confrontos, limitado à janela especificada.
                historico = historico_confrontos[key][-janela:]

            vitorias_casa = 0
            empates = 0
            derrotas_casa = 0
            for resultado in historico:
                # Ajusta a contagem com base em quem jogou em casa *NAQUELE* jogo.
                if resultado == 'H':
                    if key[0] == time_casa:  # Time casa ganhou
                        vitorias_casa +=1
                    else:                   # Time visitante ganhou
                        derrotas_casa += 1

                elif resultado == 'A':
                    if key[0] == time_casa:  #Time casa perdeu
                        derrotas_casa += 1
                    else:                     #Time visitante ganhou
                        vitorias_casa += 1
                else:
                    empates += 1  # Empate não muda, independente de quem jogou em casa

            resultados_casa.append(f"{vitorias_casa}-{empates}-{derrotas_casa}")
            resultados_visitante.append(f"{len(historico)-vitorias_casa-empates}-{empates}-{vitorias_casa}")


            # Adiciona o resultado *deste* jogo ao histórico (para uso futuro).
            historico_confrontos.setdefault(key, []).append(row['Resultado']) #.setdefault() evita erros caso a chave não exista

        df['Confronto_Direto_Casa'] = resultados_casa
        df['Confronto_Direto_Visitante'] = resultados_visitante
        return df
      
    df = calcular_confronto_direto(df)
    # --- Probabilidades Implícitas ---

    def calcular_probabilidades_implicitas(df, colunas_odds):
        """Calcula as probabilidades implícitas a partir das odds (1 / odd)."""
        for col in colunas_odds:
            # Trata o caso de divisão por zero, atribuindo uma probabilidade pequena (0.0001)
            df[f'Prob_{col}'] = df[col].apply(lambda x: 1/x if (x != 0 and not pd.isna(x)) else 0.0001)

        # Exemplo: Diferença de probabilidade entre casa e visitante (usando Bet365)
        df['Prob_Diff'] = df['Prob_B365_Casa'] - df['Prob_B365_Visitante']
        return df

    df = calcular_probabilidades_implicitas(df, colunas_odds)

    # --- Label Encoding do Resultado (ANTES do One-Hot Encoding) ---
    resultado_mapping = {'H': 0, 'D': 1, 'A': 2}  # Dicionário de mapeamento
    df['Resultado_Numerico'] = df['Resultado'].map(resultado_mapping) # Nova coluna numérica

    # --- Elo Rating ---
    def calcular_elo(df, k=32, initial_rating=1500):
        """
        Calcula o Elo rating para cada time.

        Args:
            df: DataFrame com os dados dos jogos.
            k: O fator K (constante de atualização do Elo).
            initial_rating: Rating inicial para todos os times.
        Returns:
            DataFrame com as colunas 'Elo_Casa' e 'Elo_Visitante' adicionadas.
        """
        ratings = {}  # Dicionário para armazenar os ratings: {time: rating}
        elo_casa = []
        elo_visitante = []

        for _, row in df.iterrows():
            time1 = row['TimeCasa']
            time2 = row['TimeVisitante']

            # Obtém os ratings atuais, usando o rating inicial se não houver.
            rating1 = ratings.get(time1, initial_rating)
            rating2 = ratings.get(time2, initial_rating)

            # Calcula a probabilidade de vitória esperada (Equação do Elo).
            resultado_esperado1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
            #resultado_esperado2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400)) #já calcula 1- resultado_esperado1

            # Obtém o resultado real do jogo (0: derrota, 1: vitória, 0.5: empate).
            if row['Resultado'] == 'H':
                resultado_real1 = 1
            elif row['Resultado'] == 'A':
                resultado_real1 = 0
            else:
                resultado_real1 = 0.5
            resultado_real2 = 1 - resultado_real1 # O resultado do visitante é o oposto.

            # Atualiza os ratings.
            novo_rating1 = rating1 + k * (resultado_real1 - resultado_esperado1)
            novo_rating2 = rating2 + k * (resultado_real2 - (1 - resultado_esperado1))

            # Armazena os ratings.
            ratings[time1] = novo_rating1
            ratings[time2] = novo_rating2

            elo_casa.append(rating1) # Antes da atualização
            elo_visitante.append(rating2) # Antes da atualização


        df['Elo_Casa'] = elo_casa
        df['Elo_Visitante'] = elo_visitante

        return df
    df = calcular_elo(df)

    # --- Criação das colunas Over/Under ---  NOVO!
    limites_gols = [0.5, 1.5, 2.5, 3.5] # Defina aqui os seus limites.

    for limite in limites_gols:
        df[f'Over_{limite}'] = (df['TotalGols'] > limite).astype(int)
        df[f'Under_{limite}'] = (df['TotalGols'] <= limite).astype(int)


    # --- One-Hot Encoding ---

    # Lista de colunas categóricas que precisam de encoding (agora sem 'Resultado')
    colunas_categoricas = ['Divisao', 'TimeCasa', 'TimeVisitante', 'ResultadoHT', 'Arbitro', 'DesempenhoCasa', 'DesempenhoFora','Confronto_Direto_Casa','Confronto_Direto_Visitante']
    df = pd.get_dummies(df, columns=colunas_categoricas, prefix=colunas_categoricas, dummy_na=False)  # dummy_na=False: não cria colunas para NaN

    # --- Divisão em Treino e Teste (Preparação para a Modelagem) ---

    # Importante: Ordenar por data antes de dividir!
    df = df.sort_values('Data')

    #Removendo NAs eventuais
    df.dropna(inplace=True)

    # Dividir 80% para treino, 20% para teste
    # test_size=0.2 define a proporção do conjunto de teste (20%)
    # random_state=42 garante que a divisão seja sempre a mesma (reprodutibilidade)
    # shuffle=False  Desativa a mistura. Os dados já estão ordenados por data.
    from sklearn.model_selection import train_test_split
    df_treino, df_teste = train_test_split(df, test_size=0.2, shuffle=False, random_state=42)

    print("\nTamanho do DataFrame de Treino:", len(df_treino))
    print("Tamanho do DataFrame de Teste:", len(df_teste))

    # Exibir as primeiras linhas do conjunto de treino
    print("\nDataFrame de Treino (primeiras linhas):")
    print(df_treino.head())

    # Exibir as primeiras linhas do conjunto de teste
    print("\nDataFrame de Teste (primeiras linhas):")
    print(df_teste.head())

     # --- Preparação para a Modelagem: Separar Features (X) e Target (y) ---

    # 1. Para prever o RESULTADO (H/D/A):

    # Todas as colunas, exceto as originais de resultado, data, hora, gols e a nova coluna numérica do resultado
    features_resultado = [col for col in df.columns if col not in ['Resultado', 'ResultadoHT','Data', 'Hora', 'GolsCasa', 'GolsVisitante', 'TotalGols', 'Resultado_Numerico']]

    X_treino_resultado = df_treino[features_resultado]
    # y_treino_resultado = df_treino['Resultado']  # A coluna 'Resultado' é o nosso alvo -- REMOVIDO
    X_teste_resultado = df_teste[features_resultado]
    # y_teste_resultado = df_teste['Resultado']       -- REMOVIDO


    # --- Label Encoding do Resultado (H=0, D=1, A=2) ---  REMOVIDO, já feito antes
    #resultado_mapping = {'H': 0, 'D': 1, 'A': 2}
    #df_treino['Resultado_Numerico'] = df_treino['Resultado'].map(resultado_mapping)
    #df_teste['Resultado_Numerico'] = df_teste['Resultado'].map(resultado_mapping)

    #Agora, o target é a nova coluna numérica
    y_treino_resultado = df_treino['Resultado_Numerico']
    y_teste_resultado =  df_teste['Resultado_Numerico']

    # --- (O restante do código permanece o mesmo) ---
    #X_teste_resultado = df_teste[features_resultado] #Mantém, só o target mudou

    # 2. Para prever o TOTAL DE GOLS: REMOVIDO

    #Usaremos as mesmas features, mas o target agora é 'TotalGols'
    #features_gols = features_resultado # Mesmas features
    #X_treino_gols = df_treino[features_gols]
    #y_treino_gols = df_treino['TotalGols']  # 'TotalGols' é o alvo
    #X_teste_gols = df_teste[features_gols]
    #y_teste_gols = df_teste['TotalGols']

    #Verificação de Nans:
    print("\nNaNs em X_treino_resultado:", X_treino_resultado.isna().sum().sum())
    print("NaNs em y_treino_resultado:", y_treino_resultado.isna().sum().sum())
    print("NaNs em X_teste_resultado:", X_teste_resultado.isna().sum().sum())
    print("NaNs em y_teste_resultado:", y_teste_resultado.isna().sum().sum())

    #print("\nNaNs em X_treino_gols:", X_treino_gols.isna().sum().sum())
    #print("NaNs em y_treino_gols:", y_treino_gols.isna().sum().sum())
    #print("NaNs em X_teste_gols:", X_teste_gols.isna().sum().sum())
    #print("NaNs em y_teste_gols:", y_teste_gols.isna().sum().sum())


    # --- Treinamento dos Modelos ---

    # 1. Modelo para prever o RESULTADO (Regressão Logística):
    modelo_resultado = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')  #Aumentei max_iter e alterei o solver
    modelo_resultado.fit(X_treino_resultado, y_treino_resultado)

    # 2. Modelo para prever o TOTAL DE GOLS (Regressão Linear): REMOVIDO
    #modelo_gols = LinearRegression()
    #modelo_gols.fit(X_treino_gols, y_treino_gols)


    # --- Previsões ---

    previsoes_resultado = modelo_resultado.predict(X_teste_resultado)  # Previsões nos dados de teste
    #previsoes_gols = modelo_gols.predict(X_teste_gols)  # Previsões de total de gols REMOVIDO


    # --- Avaliação dos Modelos ---

    # 1. Avaliação do modelo de previsão de RESULTADO:

    print("\n--- Avaliação do Modelo de Previsão de Resultado ---")
    print("Acurácia:", accuracy_score(y_teste_resultado, previsoes_resultado))
    print(classification_report(y_teste_resultado, previsoes_resultado)) # Relatório completo
    print("Matriz de Confusão:\n", confusion_matrix(y_teste_resultado, previsoes_resultado))


    # 2. Avaliação do modelo de previsão de TOTAL DE GOLS: REMOVIDA

    #print("\n--- Avaliação do Modelo de Previsão de Total de Gols ---")
    #rmse = mean_squared_error(y_teste_gols, previsoes_gols)  # Calcula o RMSE.  Removemos squared=False
    #print("RMSE (Raiz do Erro Quadrático Médio):", rmse)

     # --- Criação das colunas Over/Under ---  NOVO!
    limites_gols = [0.5, 1.5, 2.5, 3.5] # Defina aqui os seus limites.

    for limite in limites_gols:
        df[f'Over_{limite}'] = (df['TotalGols'] > limite).astype(int)
        df[f'Under_{limite}'] = (df['TotalGols'] <= limite).astype(int)

    # --- Treinamento e Avaliação dos Modelos Over/Under (Regressão Logística) ---
    #Após a divisão em treino e teste:

    for limite in limites_gols:
        print(f"\n\n--- Treinando e Avaliando Modelo para Over/Under {limite} Gols ---")

        # 1. Preparar os dados (X e y) para *este* limite:
        features = [col for col in df.columns if col not in ['Resultado', 'ResultadoHT','Data', 'Hora', 'GolsCasa', 'GolsVisitante', 'TotalGols', 'Resultado_Numerico'] and not col.startswith('Under_') and not col.startswith('Over_')]  #Só as features
        X_treino = df_treino[features]
        y_treino = df_treino[f'Over_{limite}']  # Over/Under como target.
        X_teste = df_teste[features]
        y_teste = df_teste[f'Over_{limite}']

        #Verificação de Nans:
        print("\nNaNs em X_treino:", X_treino.isna().sum().sum())
        print("NaNs em y_treino:", y_treino.isna().sum().sum())
        print("NaNs em X_teste:", X_teste.isna().sum().sum())
        print("NaNs em y_teste:", y_teste.isna().sum().sum())
        # 2. Treinar o modelo (Regressão Logística):
        modelo = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
        modelo.fit(X_treino, y_treino)

        # 3. Fazer previsões:
        previsoes = modelo.predict(X_teste)

        # 4. Avaliar:
        print(f"\n--- Avaliação do Modelo Over/Under {limite} ---")
        print("Acurácia:", accuracy_score(y_teste, previsoes))
        print(classification_report(y_teste, previsoes))
        print("Matriz de Confusão:\n", confusion_matrix(y_teste, previsoes))

else:
    print("Falha ao obter ou analisar o CSV.")