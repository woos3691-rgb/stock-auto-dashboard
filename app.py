import io
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='쉬운 주식 분석기', page_icon='📈', layout='wide')
st.title('📈 쉬운 주식 분석기')
st.caption('종목코드 하나로 주가 · RSI(14) · 거래량 배수 · 가격대별 거래량을 확인합니다. 매매 추천이 아닌 정보 제공용입니다.')

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(code: str, period: str):
    code = code.strip().upper()
    if re.fullmatch(r'\d{6}', code):
        choices = [code + '.KS', code + '.KQ']
    else:
        choices = [code]
    failures = []
    for ticker in choices:
        try:
            frame = yf.Ticker(ticker).history(period=period, interval='1d', auto_adjust=True, timeout=18)
            if frame is not None and len(frame) >= 35:
                frame = frame[['Open','High','Low','Close','Volume']].copy()
                frame = frame.dropna(subset=['Open','High','Low','Close'])
                frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
                return ticker, frame, ''
            failures.append(ticker + ': 데이터가 충분하지 않음')
        except Exception as exc:
            failures.append(ticker + ': ' + str(exc)[:130])
    return '', pd.DataFrame(), ' / '.join(failures)

def rsi_14(closes):
    diffs = closes.diff()
    upward = diffs.clip(lower=0)
    downward = (-diffs).clip(lower=0)
    gains = upward.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    losses = downward.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((losses == 0) & (gains > 0), 100)
    result = result.mask((losses == 0) & (gains == 0), 50)
    return result

def volume_profile(prices, bins=24):
    # Approximation: each day's entire volume is assigned to a typical price; not exact tick-level VPVR.
    tp = ((prices['High'] + prices['Low'] + prices['Close'])/3).to_numpy(dtype=float)
    v = prices['Volume'].fillna(0).to_numpy(dtype=float)
    hist, edges = np.histogram(tp, bins=bins, weights=v)
    return hist, (edges[:-1]+edges[1:])/2

with st.form('search'):
    c1, c2, c3 = st.columns([2,1,1])
    code = c1.text_input('종목코드 (한국은 숫자 6자리)', value='005930', placeholder='예: 005930 또는 AAPL')
    period = c2.selectbox('분석 기간', ['6mo','1y','2y'], index=1, format_func=lambda k: {'6mo':'6개월','1y':'1년','2y':'2년'}[k])
    go_button = c3.form_submit_button('🔍 분석 시작', use_container_width=True)
st.caption('예: 삼성전자 005930 · SK하이닉스 000660 · 미국 애플 AAPL')

if not code.strip():
    st.info('종목코드를 입력하고 분석 시작을 누르세요.')
    st.stop()
if not re.fullmatch(r'(\d{6}|[A-Za-z][A-Za-z0-9.\-]{0,14})', code.strip()):
    st.error('한국 주식은 6자리 숫자, 미국 주식은 AAPL처럼 영문 종목코드를 입력하세요.')
    st.stop()

with st.spinner('실제 주가 데이터를 가져오는 중입니다. 잠시 기다려 주세요…'):
    ticker, df, problem = fetch_prices(code, period)
if df.empty:
    st.error('주가를 가져오지 못했습니다. 잠시 후 다시 시도하거나 다른 종목코드를 확인해 주세요.')
    with st.expander('조회 오류 자세히 보기'):
        st.code(problem)
    st.caption('무료 데이터 제공처의 일시적 제한 또는 네트워크 문제일 수 있습니다.')
    st.stop()

close = df['Close']
df['등락률(%)'] = close.pct_change() * 100
df['RSI(14)'] = rsi_14(close)
df['거래량 20일 평균'] = df['Volume'].shift(1).rolling(20, min_periods=20).mean()
df['거래량 배수'] = df['Volume'] / df['거래량 20일 평균'].replace(0,np.nan)
df['이동평균 20일'] = close.rolling(20).mean()
df['이동평균 60일'] = close.rolling(60).mean()
last = df.iloc[-1]
previous = df.iloc[-2]
is_kr = ticker.endswith(('.KS','.KQ'))
currency = '원' if is_kr else '달러'
fmt = lambda n: f'{n:,.0f}' if is_kr else f'{n:,.2f}'

st.success(f'조회 성공 · {ticker} · 마지막 거래일 {df.index[-1].date()} · 일봉 기준')
if (pd.Timestamp.now().normalize() - df.index[-1]).days > 7:
    st.warning('마지막 거래일이 오래되었습니다. 데이터 갱신 상태를 확인하세요.')

x1,x2,x3,x4 = st.columns(4)
x1.metric('최근 종가', f'{fmt(last["Close"])} {currency}', f'{last["등락률(%)"]:+.2f}%' if pd.notna(last['등락률(%)']) else None)
x2.metric('RSI(14)', f'{last["RSI(14)"]:.1f}' if pd.notna(last['RSI(14)']) else '계산 중')
x3.metric('거래량 배수', f'{last["거래량 배수"]:.2f}배' if pd.notna(last['거래량 배수']) else '계산 중')
x4.metric('최근 거래량', f'{int(last["Volume"]):,}주')
st.caption('거래량 배수는 오늘 거래량 ÷ 직전 20거래일 평균 거래량입니다. 일봉 기준으로 계산합니다.')

fig = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.035,row_heights=[0.55,0.23,0.22],subplot_titles=('주가와 이동평균','거래량','RSI(14)'))
fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name='주가'),row=1,col=1)
for field,name in [('이동평균 20일','20일 평균'),('이동평균 60일','60일 평균')]:
    fig.add_trace(go.Scatter(x=df.index,y=df[field],mode='lines',name=name),row=1,col=1)
fig.add_trace(go.Bar(x=df.index,y=df.Volume,name='거래량',marker_color=np.where(df.Close >= df.Open,'#e15b56','#438ec8')),row=2,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df['RSI(14)'],mode='lines',name='RSI(14)'),row=3,col=1)
fig.add_hline(y=70,row=3,col=1,line_dash='dot',line_color='#c95454')
fig.add_hline(y=30,row=3,col=1,line_dash='dot',line_color='#477ed0')
fig.update_layout(height=750,xaxis_rangeslider_visible=False,hovermode='x unified',legend=dict(orientation='h',y=1.05),margin=dict(l=20,r=20,t=70,b=20))
fig.update_yaxes(range=[0,100],row=3,col=1)
st.plotly_chart(fig,use_container_width=True)

st.subheader('📊 추정 매물대 (가격대별 거래량)')
hist, centers = volume_profile(df)
profile = go.Figure(go.Bar(x=hist,y=centers,orientation='h',marker_color='#537fad',customdata=np.asarray(hist).reshape(-1,1),hovertemplate='가격: %{y:,.2f}<br>추정 거래량: %{x:,.0f}주<extra></extra>'))
profile.add_hline(y=float(last.Close),line_color='#f97316',line_dash='dash',annotation_text='최근 종가')
profile.update_layout(height=440,xaxis_title='거래량(주)',yaxis_title=f'가격({currency})',margin=dict(l=15,r=15,t=15,b=15))
st.plotly_chart(profile,use_container_width=True)
st.info('매물대는 일별 고가·저가·종가의 평균가격에 해당 날짜 거래량을 배분한 **추정치**입니다. 실제 체결가격별 매물대와 다를 수 있습니다.')

st.subheader('🔎 숫자 쉽게 읽기')
rsi = float(last['RSI(14)'])
mult = float(last['거래량 배수'])
if rsi >= 70:
    st.write(f'• RSI {rsi:.1f}: 최근 상승세가 강하고 과열 구간일 수 있습니다. 바로 하락한다는 의미는 아닙니다.')
elif rsi <= 30:
    st.write(f'• RSI {rsi:.1f}: 최근 하락세가 강하고 과매도 구간일 수 있습니다. 바로 상승한다는 의미는 아닙니다.')
else:
    st.write(f'• RSI {rsi:.1f}: 관례적인 과매수(70 이상)·과매도(30 이하) 범위 밖이 아닙니다.')
if mult >= 3:
    st.write(f'• 거래량 {mult:.2f}배: 직전 20거래일 평균보다 거래가 크게 늘었습니다. 상승과 하락 어느 쪽에서도 발생합니다.')
else:
    st.write(f'• 거래량 {mult:.2f}배: 직전 20거래일 평균의 약 {mult:.2f}배입니다.')
st.write('• 매물대가 두꺼운 가격은 과거 거래가 집중된 구간입니다. 반드시 지지선이나 저항선이 되는 것은 아닙니다.')

st.subheader('📋 날짜별 데이터')
view = df[['Open','High','Low','Close','Volume','등락률(%)','RSI(14)','거래량 배수']].copy()
view.columns = ['시가','고가','저가','종가','거래량','등락률(%)','RSI(14)','거래량 배수']
view.index.name='날짜'
st.dataframe(view.iloc[::-1].round(2),use_container_width=True)
csv = view.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
st.download_button('⬇️ 분석 결과 엑셀용 CSV 저장',csv,file_name=f'{ticker}_주식분석.csv',mime='text/csv')
st.divider()
st.caption('데이터 출처: Yahoo Finance (yfinance 비공식 접속). 보통 지연 데이터이며 실시간 주문용 시세가 아닙니다. 무료 제공 정책 또는 접속 제한에 따라 조회가 실패할 수 있습니다. 투자 판단 및 손실의 책임은 이용자에게 있습니다.')
