import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

# Configuração da página
st.set_page_config(
    page_title='Projeto 3', 
    page_icon='📊', 
    layout='wide'
)

# Descrição do projeto
st.header('Projeto 3')
st.markdown('**Detecção de *outliers* nos retornos percentuais de um ativo, com IsolationForest.**')

# Função para calcular retornos e detectar outliers
def detect_outliers(data, contamination):
    # Calcula os retornos diários
    data['retornos'] = data['Close'].pct_change()

    # Ajustar dados para detecção de outliers, removendo NaNs
    returns = data['retornos'].dropna().values.reshape(-1, 1)

    # Aplicar Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=1)
    outliers = model.fit_predict(returns)
    data['outlier'] = pd.Series([1] + list(outliers), index=data.index)

    return data

# Configuração da barra lateral
st.sidebar.header('Menu de opções')

# Seleção do ticker
selected_stock = st.sidebar.text_input('Digite o ticker da ação', value='BTC-USD').upper()

# Seleção das datas de início e fim
start_date = st.sidebar.date_input('Data inicial', value=pd.to_datetime('today') - pd.DateOffset(days=100))
end_date = st.sidebar.date_input('Data final', value=pd.to_datetime('today'))

# Slider para ajustar o parâmetro contamination
contamination = st.sidebar.slider(
    'Sensibilitdade da detecção',
    min_value=0.01,
    max_value=0.5,
    value=0.1, 
    step=0.01
)

# Baixa os dados da ação selecionada
data = yf.Ticker(selected_stock).history(start=start_date, end=end_date)

if not data.empty:
    # Detecta os outliers
    data = detect_outliers(data, contamination)

    # Cria os subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Retornos', 'Preço de Fechamento'),
        vertical_spacing=0.2  # Espaçamento entre os subgráficos
    )

    # Gráfico de retornos
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['retornos'], 
        mode='lines', 
        name='retornos',
        line=dict(color='orange', width=2)
    ), row=1, col=1)

    # Destaca os outliers no gráfico de retornos
    outliers_returns = data[data['outlier'] == -1]
    fig.add_trace(go.Scatter(
        x=outliers_returns.index, 
        y=outliers_returns['retornos'], 
        mode='markers', 
        name='Outliers', 
        marker=dict(color='red', size=10, symbol='x')
    ), row=1, col=1)

    # Gráfico de preço de fechamento
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines', 
        name='Preço de Fechamento',
        line=dict(color='darkblue', width=2)
    ), row=2, col=1)

    # Destaca os outliers no gráfico de preço
    outliers_price = data[data['outlier'] == -1]
    fig.add_trace(go.Scatter(
        x=outliers_price.index, 
        y=outliers_price['Close'], 
        mode='markers', 
        name='Outliers', 
        marker=dict(color='red', size=10, symbol='x')
    ), row=2, col=1)

    # Configura o layout do gráfico
    fig.update_layout(
        height=700,
        xaxis_title='Data',
        hovermode='x unified', # Unifica o tooltip do eixo x para os dois gráficos
        legend=dict(
            orientation='h',  
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Ajusta os eixos y do subplot de retornos
    fig.update_yaxes(title_text='Retornos', row=1, col=1)

    # Habilita a vinculação do eixo x
    fig.update_xaxes(matches='x')

    # Exibe o gráfico
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f'Não foram encontrados dados para o ativo {selected_stock}.')

st.sidebar.markdown('''
    <p style="margin-top: 30px; text-align: center">
        Projetos Python para o Mercado Financeiro<br>
        <a href="https://www.instagram.com/cientistadabolsa">@cientistadabolsa</a>
    </p>
''', unsafe_allow_html=True)

