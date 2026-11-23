(() => {
  const root=document.getElementById('stock-dashboard');
  const bootstrap=globalThis.dashboardPreview||null;
  const slot=name=>root.querySelector(`[data-slot="${name}"]`);
  const fmt=(v,d=2)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
  const sign=v=>(v>=0?'+':'')+fmt(v);
  const scaled=(v,divisor)=>v==null?null:v/divisor;
  const escape=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
  const tone=v=>v>0?'sd-up':v<0?'sd-down':'';
  const stored=(key,fallback)=>{try{return JSON.parse(localStorage.getItem(key))||fallback;}catch{return fallback;}};
  const persist=(key,value)=>{try{localStorage.setItem(key,JSON.stringify(value));}catch{}};
  const state={meta:null,market:null,stock:null,code:'002957.SZ',asOf:'',board:'all',sort:'amount',desc:true,page:1,
    list:'all',recent:stored('quant-recent',[]),period:'day',count:bootstrap?60:100,adjustment:'none',
    sub:'amount',pageName:'market',query:stored('quant-stock-query',''),request:0};
  let marketRequest=0,searchRequest=0,searchTimer;
  const indexNames={'000001.SH':'上证指数','000300.SH':'沪深300','000905.SH':'中证500','000852.SH':'中证1000'};
  const offlineRead=(url)=>{
    const parsed=new URL(url,'http://preview.local'),params=parsed.searchParams,path=parsed.pathname;
    if(path==='/api/meta')return bootstrap.meta;
    if(path==='/api/stocks/search'){
      const q=params.get('q').trim().toLowerCase();
      const rows=bootstrap.catalog.filter(s=>[s.ts_code,s.symbol,s.name,s.industry].some(v=>String(v||'').toLowerCase().includes(q)));
      rows.sort((a,b)=>(a.ts_code===q||a.symbol===q?-1:0)-(b.ts_code===q||b.symbol===q?-1:0)||a.ts_code.localeCompare(b.ts_code));
      return {total:rows.length,results:rows.slice(0,20)};
    }
    if(path==='/api/market'){
      const board=params.get('market'),codes=params.get('codes'),sort=params.get('sort'),desc=params.get('desc')==='true',page=Number(params.get('page'));
      let rows=bootstrap.market.results.slice();
      if(board!=='all')rows=rows.filter(s=>s.market===({main:'主板',gem:'创业板',star:'科创板',bj:'北交所'})[board]);
      if(codes)rows=rows.filter(s=>codes.split(',').includes(s.ts_code));
      const field={amount:'amount_yi',change:'pct_chg',price:'close',turnover:'turnover_rate',code:'ts_code'}[sort];
      rows.sort((a,b)=>{const av=a[field],bv=b[field];if(av==null)return 1;if(bv==null)return -1;const order=typeof av==='number'?av-bv:String(av).localeCompare(String(bv));return(desc?-1:1)*order||a.ts_code.localeCompare(b.ts_code);});
      return {...bootstrap.market,total:rows.length,page,page_size:10,results:rows.slice((page-1)*10,page*10)};
    }
    if(path.startsWith('/api/stocks/')&&path.endsWith('/bars')){
      const code=path.split('/')[3],key=[code,params.get('period'),params.get('adjustment')].join('|');
      const body=bootstrap.charts[key];
      if(!body)throw new Error('此内嵌预览未载入该股票图表，请打开本地应用查看完整行情');
      return {...body,bars:body.bars.slice(-Number(params.get('count')||100))};
    }
    if(path.startsWith('/api/modules/')){
      if(path.endsWith('/features')){
        const data=bootstrap.features[params.get('code')];
        if(!data)throw new Error('对话预览未载入该股票特征，请打开本地应用查询');
        return data;
      }
      return bootstrap.modules;
    }
    throw new Error('预览中未提供此记录');
  };
  async function read(url){
    if(bootstrap)return offlineRead(url);
    const response=await fetch(url);
    const body=await response.json();
    if(!response.ok)throw new Error(body.detail||'请求失败');
    return body;
  }
  function status(message){slot('app-status').textContent=message;}
  function saveUI(){
    if(window.openai?.setWidgetState)window.openai.setWidgetState({modelContent:{page:state.pageName,stock:state.code,
      period:state.period,adjustment:state.adjustment,feature:research.key,model:research.model,
      signalDate:research.signalDate,rule:research.rule,modelPeriod:research.period,budget:research.budget}}).catch(()=>{});
  }
  async function loadMarket(){
    const request=++marketRequest;
      const codes=state.list==='recent'?state.recent:null;
    try{
      const params=new URLSearchParams({as_of:state.asOf,market:state.board,sort:state.sort,desc:String(state.desc),page:String(state.page),page_size:'10'});
      if(codes)params.set('codes',codes.length?codes.join(','):'__empty__');
      const body=await read('/api/market?'+params);
      if(request!==marketRequest)return;
      state.market=body;renderMarket();status('');
    }catch(error){if(request===marketRequest)status(error.message);}
  }
  function renderMarket(){
    const body=state.market;
    slot('indexes').innerHTML=body.indexes.map(i=>`<div class="sd-index"><label>${indexNames[i.ts_code]||i.name}</label><strong>${fmt(i.close)}</strong><span class="sd-change ${tone(i.pct_chg)}">${sign(i.pct_chg)}%</span></div>`).join('');
    const b=body.breadth;
    slot('breadth').innerHTML=`<span>涨跌分布</span><div class="sd-breadth-plot" role="img" aria-label="上涨${b.rising}，下跌${b.falling}，平盘${b.unchanged}"><span style="width:${b.count?b.rising/b.count*100:0}%;background:var(--sd-red)"></span><span style="width:${b.count?b.falling/b.count*100:0}%;background:var(--sd-green)"></span><span style="width:${b.count?b.unchanged/b.count*100:0}%;background:var(--sd-line)"></span></div><span class="sd-up">涨 ${fmt(b.rising,0)}</span><span class="sd-down">跌 ${fmt(b.falling,0)}</span><span>平 ${fmt(b.unchanged,0)}</span><span class="sd-market-amount">成交额 ${fmt(b.amount_yi,0)} 亿元</span>`;
    renderMarketInsight();
    slot('list-count').textContent=fmt(body.total,0)+'只';
    slot('stocks').innerHTML=body.results.length?body.results.map(s=>`<tr aria-selected="${s.ts_code===state.code}"><td><button type="button" data-stock="${s.ts_code}" data-stock-name="${escape(s.name||s.ts_code)}">${escape(s.name||s.ts_code)}<small>${s.symbol||s.ts_code.split('.')[0]}</small></button></td><td>${fmt(s.close)}</td><td class="${tone(s.pct_chg)}">${sign(s.pct_chg)}%</td><td>${fmt(s.amount_yi)}亿</td></tr>`).join(''):'<tr><td colspan="4" style="text-align:center;padding:35px">暂无股票</td></tr>';
    slot('pagination').textContent=`${body.page} / ${Math.max(1,Math.ceil(body.total/body.page_size))}`;
    root.querySelector('[data-action="previous"]').disabled=body.page<=1;
    root.querySelector('[data-action="next"]').disabled=body.page*body.page_size>=body.total;
  }
  function clearFeatureDetails(message){
    slot('feature-title').textContent=message;
    slot('feature-date').textContent=`行情截面 ${state.asOf||'—'} · ${featureStockName()}`;
    slot('feature-metrics').replaceChildren();
    slot('feature-formula').textContent='';
    slot('feature-insight').textContent=message;
    slot('attribution-method').textContent='';
    slot('attribution-summary').replaceChildren();
    slot('attribution-factors').replaceChildren();
    slot('attribution-caption').textContent='';
    root.querySelectorAll('[data-research-chart="distribution"],[data-research-chart="feature-line"]').forEach(chart=>chart.replaceChildren());
    slot('sequence-count').textContent='';
    slot('sequence-summary').textContent=message;
    slot('sequence-description').textContent=message;
    slot('sequence-head').replaceChildren();
    slot('sequence-body').replaceChildren();
  }
  function renderStockLoading(name,code){
    slot('stock-name').textContent=name||code;
    slot('stock-code').textContent=code;
    slot('price').textContent='—';
    slot('price').className='';
    slot('change').textContent='正在读取行情…';
    slot('change').className='';
    slot('quote-date').textContent='';
    slot('quote-fields').replaceChildren();
    slot('chart-readout').textContent='正在读取行情…';
    root.querySelectorAll('[data-chart="price"],[data-chart="sub"]').forEach(chart=>chart.replaceChildren());
    renderMarketInsight();
  }
  function recentStockName(code){
    return research.recent.find(item=>item.code===code)?.name||
      state.market?.results.find(item=>item.ts_code===code)?.name||
      bootstrap?.catalog.find(item=>item.ts_code===code)?.name||code;
  }
  async function loadStock(code=state.code,selectedName=''){
    const request=++state.request;
    const stockChanged=research.stockCode!==code;
    const name=selectedName||recentStockName(code);
    state.code=code;
    research.stockCode=code;
    research.stockName=name;
    research.stockError=false;
    if(stockChanged){
      state.stock=null;
      research.features=null;research.sequence=null;research.data=null;
      ++research.request;++research.sequenceRequest;
    }
    if(stockChanged||!state.stock){
      research.loadingStock=true;
      renderStockLoading(name,code);
      if(state.market)renderMarket();
      clearFeatureDetails(`正在读取 ${name} 的量价特征…`);
      renderFeatureIntro();
    }
    const loadFeatureDuringBars=(stockChanged||!state.stock)&&state.pageName==='features';
    status('正在读取行情…');
    if(loadFeatureDuringBars)loadResearch();
    try{
      const params=new URLSearchParams({as_of:state.asOf,period:state.period,adjustment:state.adjustment,count:String(state.count)});
      const body=await read('/api/stocks/'+encodeURIComponent(code)+'/bars?'+params);
      if(request!==state.request)return;
      state.code=code;state.stock=body;
      research.loadingStock=false;
      state.recent=[code,...state.recent.filter(c=>c!==code)].slice(0,20);
      persist('quant-recent',state.recent);
      research.stockCode=code;research.stockName=body.name||code;
      research.recent=[{code,name:research.stockName},...research.recent.filter(item=>item.code!==code)].slice(0,6);
      persist('quant-feature-recent',research.recent);renderRecentResearch();renderFeatureIntro();
      renderStock();if(state.list==='recent')loadMarket();else if(state.market)renderMarket();status('');saveUI();
      if(state.pageName!=='market'&&!(loadFeatureDuringBars&&state.pageName==='features'))loadResearch();
    }catch(error){
      if(request!==state.request)return;
      research.loadingStock=false;research.stockError=true;
      renderStockLoading(name,code);
      slot('change').textContent='行情暂不可用';
      if(state.pageName==='features')clearFeatureDetails(`${name} 的量价数据暂不可用`);
      renderFeatureIntro();status(error.message);
    }
  }
  function renderStock(){
    const s=state.stock,q=s.quote;
    slot('stock-name').textContent=s.name;
    slot('stock-code').textContent=s.ts_code+'　'+s.industry;
    slot('price').textContent=fmt(q.close);slot('price').className=tone(q.pct_chg);
    slot('change').textContent=sign(q.close-q.pre_close)+'　'+sign(q.pct_chg)+'%';slot('change').className=tone(q.pct_chg);
    slot('quote-date').textContent=q.trade_date+' 收盘';
    slot('quote-fields').innerHTML=[
      ['今开',fmt(q.open)],['最高',fmt(q.high)],['最低',fmt(q.low)],['昨收',fmt(q.pre_close)],
      ['换手',fmt(q.turnover_rate)+'%'],['成交额',fmt(scaled(q.amount,1e5))+'亿'],
      ['总市值',fmt(scaled(q.total_mv,1e4))+'亿'],['流通市值',fmt(scaled(q.circ_mv,1e4))+'亿']
    ].map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    renderMarketInsight();
    renderCharts();
  }
  function renderMarketInsight(){
    const market=state.market,stock=state.stock;
    if(!market)return;
    const breadth=market.breadth||{},net=(breadth.rising||0)-(breadth.falling||0);
    const breadthText=net>0?`上涨多于下跌 ${fmt(net,0)} 只`:net<0?`下跌多于上涨 ${fmt(Math.abs(net),0)} 只`:'上涨与下跌家数相同';
    const indexes=(market.indexes||[]).filter(item=>Number.isFinite(Number(item.pct_chg)));
    const indexText=indexes.length?`宽基指数中上涨 ${indexes.filter(item=>item.pct_chg>0).length} / ${indexes.length} 个`:'';
    let stockText=research.loadingStock?`正在读取 ${research.stockName||state.code}（${state.code}）行情。`:
      research.stockError?`${research.stockName||state.code}（${state.code}）行情暂不可用。`:'选择一只股票查看其收盘变化与K线走势。';
    if(stock){
      const bars=stock.bars||[],first=bars.length>1?bars[Math.max(0,bars.length-20)]:null,last=bars.at(-1);
      const trend=first&&last&&first.close?((last.close/first.close)-1)*100:null;
      stockText=`${stock.name}（${stock.ts_code}）于 ${stock.quote.trade_date} 收盘 ${fmt(stock.quote.close)} 元，涨跌 ${sign(stock.quote.pct_chg)}%。${trend==null?'所选区间不足以比较。':`近 ${Math.min(20,bars.length)} 根${state.period==='day'?'日':state.period==='week'?'周':'月'}K 线累计 ${sign(trend)}%。`}`;
    }
    slot('market-insight').textContent=`${market.as_of} 市场${breadthText}，${indexText}；成交额 ${fmt(breadth.amount_yi,0)} 亿元。${stockText}`;
  }
  const ns='http://www.w3.org/2000/svg';
  function shape(svg,tag,attrs,text){
    const el=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!=null)el.textContent=text;
    svg.append(el);return el;
  }
  function dimensions(svg){
    const width=svg.getBoundingClientRect().width,height=parseFloat(getComputedStyle(svg).height)||250;
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${width} ${height}`);return {width,height};
  }
  function readout(bar){
    const q=bar||state.stock.bars.at(-1);
    slot('chart-readout').innerHTML=`<span>${q.trade_date}</span><span class="sd-ma5">MA5 ${fmt(q.ma5)}</span><span class="sd-ma10">MA10 ${fmt(q.ma10)}</span><span class="sd-ma20">MA20 ${fmt(q.ma20)}</span>`;
  }
  function renderCharts(){
    if(!state.stock||!root.querySelector('[data-view="market"]').getClientRects().length)return;
    const bars=state.stock.bars;if(!bars.length)return;
    const price=root.querySelector('[data-chart="price"]'),sub=root.querySelector('[data-chart="sub"]');
    const {width:w,height:h}=dimensions(price),R=55,L=3,T=12,B=h-14,plotW=w-R-L;
    if(plotW<100)return;
    const low=Math.min(...bars.map(b=>b.low)),high=Math.max(...bars.map(b=>b.high)),padding=(high-low||high*.01)*.1;
    const y=v=>T+(high+padding-v)/(high-low+2*padding)*(B-T);
    const x=i=>L+(i+.5)*plotW/bars.length;
    for(let i=0;i<4;i++){
      const value=high+padding-i*(high-low+2*padding)/3,yy=y(value);
      shape(price,'line',{x1:L,x2:w-R,y1:yy,y2:yy,stroke:'var(--sd-line)','stroke-width':1});
      shape(price,'text',{x:w-R+8,y:yy+4},fmt(value,high>100?0:2));
    }
    for(const field of ['ma5','ma10','ma20']){
      let points='';
      bars.forEach((b,i)=>{if(b[field]!=null&&b[field]>=low-padding&&b[field]<=high+padding)points+=`${x(i)},${y(b[field])} `;});
      shape(price,'polyline',{points,fill:'none',stroke:`var(--sd-${field})`,'stroke-width':1.25,'pointer-events':'none'});
    }
    const barWidth=Math.max(1,plotW/bars.length*.62);
    bars.forEach((b,i)=>{
      const color=b.close>=b.open?'var(--sd-red)':'var(--sd-green)',xx=x(i);
      const group=shape(price,'g',{'data-tooltip':`${b.trade_date} 开${fmt(b.open)} 高${fmt(b.high)} 低${fmt(b.low)} 收${fmt(b.close)}元`});
      const line=document.createElementNS(ns,'line');
      Object.entries({x1:xx,x2:xx,y1:y(b.high),y2:y(b.low),stroke:color,'stroke-width':1}).forEach(([k,v])=>line.setAttribute(k,v));group.append(line);
      const rect=document.createElementNS(ns,'rect');
      Object.entries({x:xx-barWidth/2,y:Math.min(y(b.open),y(b.close)),width:barWidth,height:Math.max(1,Math.abs(y(b.close)-y(b.open))),fill:color}).forEach(([k,v])=>rect.setAttribute(k,v));group.append(rect);
      group.addEventListener('pointerenter',()=>readout(b));
      const title=document.createElementNS(ns,'title');title.textContent=group.getAttribute('data-tooltip');group.append(title);
    });
    readout();
    const {height:sh}=dimensions(sub),ST=5,SB=sh-20,field={amount:'amount_yi',volume:'volume_wan',money:'net_mf_yi'}[state.sub];
    slot('sub-unit').textContent=state.sub==='volume'?'万股':'亿元';
    const values=bars.map(b=>b[field]).filter(v=>v!=null),vmax=Math.max(0,...values)||1,vmin=state.sub==='money'?Math.min(0,...values):0,range=vmax-vmin;
    const sy=v=>ST+(vmax-v)/range*(SB-ST),zero=sy(0);
    shape(sub,'line',{x1:L,x2:w-R,y1:zero,y2:zero,stroke:'var(--sd-line)','stroke-width':1});
    bars.forEach((b,i)=>{
      if(b[field]==null)return;
      const color=state.sub==='money'?(b[field]>=0?'var(--sd-red)':'var(--sd-green)'):(b.close>=b.open?'var(--sd-red)':'var(--sd-green)');
      const mark=shape(sub,'rect',{x:x(i)-barWidth/2,y:Math.min(sy(b[field]),zero),width:barWidth,height:Math.max(0,Math.abs(sy(b[field])-zero)),fill:color,opacity:.7,'data-tooltip':`${b.trade_date} ${fmt(b[field])}${state.sub==='volume'?'万股':'亿元'}`});
      const title=document.createElementNS(ns,'title');title.textContent=mark.getAttribute('data-tooltip');mark.append(title);
    });
    shape(sub,'text',{x:w-R+8,y:ST+8},fmt(vmax,1));shape(sub,'text',{x:w-R+8,y:SB},fmt(vmin,1));
    [0,Math.floor(bars.length/2),bars.length-1].forEach((i,j)=>shape(sub,'text',{x:j===0?L:j===2?w-R:x(i),y:sh-2,'text-anchor':j===0?'start':j===2?'end':'middle'},bars[i].trade_date.slice(2)));
  }
  function sourceDialog(){
    const meta=state.meta;
    slot('source-content').innerHTML=`<p>数据来源：<a href="${meta.source_url}" target="_blank" rel="noreferrer">Tushare Pro</a>。日行情采用 <a href="${meta.daily_docs}" target="_blank" rel="noreferrer">daily</a>，换手率与市值采用 <a href="${meta.basic_docs}" target="_blank" rel="noreferrer">daily_basic</a>；资金流与指数分别来自 moneyflow 和 index_daily。数据通过项目采集与本地缓存读取。</p><table class="sd-source-table"><thead><tr><th>数据</th><th>记录数</th><th>截止日期</th></tr></thead><tbody>${meta.tables.filter(t=>['daily','daily_basic','adj_factor','moneyflow','index_daily'].includes(t.name)).map(t=>`<tr><td>${t.name}</td><td>${fmt(t.rows,0)}</td><td>${t.date_max||'—'}</td></tr>`).join('')}</tbody></table><p>报价为所选日收盘数据，停牌日不补行情。公司名称和行业采用股票资料快照。前复权以所选股票最后有效交易日的复权因子归一化；周线、月线先聚合行情，再计算均线。金额统一显示为亿元，成交量显示为万股。</p><p>本地日行情截止 ${meta.last_date}。增量更新接口尚未启用；行情更新和研究实验版本分别管理。${bootstrap?'本对话预览包含全库股票目录与当日列表，部分股票附有图表；完整查询在本地应用中提供。':''}</p>`;
    const sync=meta.incremental||{};
    const paragraphs=slot('source-content').querySelectorAll('p');
    paragraphs[paragraphs.length-1].textContent=`历史日行情截止 ${meta.tables.find(t=>t.name==='daily')?.date_max||meta.last_date}。独立增量：${sync.published_last_date?'已发布至 '+sync.published_last_date:'尚未发布新行情'}；${sync.credential_present?'本机已配置凭据':'本机待配置 Tushare token'}。增量校验后进入行情查询，研究实验继续引用冻结历史库。${bootstrap?'对话预览包含全库目录与当日列表，部分股票附有图表与特征；完整查询在本地应用中提供。':''}`;
    root.querySelector('[data-dialog="source"]').showModal();
  }
  const research={features:null,sequence:null,data:null,key:'r20',model:'xgb_F69',rule:'DB-K30-b8',period:'test',signalDate:'',featureDate:'',stockCode:state.code,stockName:'',loadingStock:false,stockError:false,recent:stored('quant-feature-recent',[]),catalogQuery:'',catalogGroup:'all',featureView:'observations',
    budget:200000,reserve:10,count:5,industryCap:2,fee:8,agentCode:null,holdings:[],blocked:[],request:0,sequenceRequest:0};
  const featureCache=new Map(),sequenceCache=new Map(),moduleCache=new Map();
  const debateOrigin=location.origin;
  const debateBase=new URL('/debate/',location.origin).href;
  let debateSyncRequest=0,debateReady=false,debateTab='stock',pendingChatPrefill='';
  const colors=['var(--sd-series-1)','var(--sd-series-2)','var(--sd-series-3)','var(--sd-series-4)','var(--sd-muted)'];
  const modelChartColors=['var(--sd-series-1)','var(--sd-series-2)','var(--sd-series-3)','var(--sd-series-4)','#303030','#999999','#5f5f5f'];
  const modelChartDashes=['none','6 2','2 2','5 3','10 3','2 3','8 2 2 2'];
  const MODEL_PEERS=['xgb_F69','rf_F69','lstm_SF69'];
  const MODEL_NAMES={xgb_F69:'XGBoost · 69 项特征',rf_F69:'随机森林 · 69 项特征',lstm_SF69:'LSTM · 69 项特征 + 60×11 序列'};
  function modelDisplayName(id,label=''){
    if(MODEL_NAMES[id])return MODEL_NAMES[id];
    const value=String(label||'');
    if(/lstm/i.test(value))return MODEL_NAMES.lstm_SF69;
    if(/random.?forest|随机森林/i.test(value))return MODEL_NAMES.rf_F69;
    if(/xgboost/i.test(value))return MODEL_NAMES.xgb_F69;
    return value;
  }
  function syncResearchControls(){
    const available=state.meta?.model_data?.available===true;
    root.querySelectorAll('[data-research-control="signal-date"]').forEach(input=>{
      input.min=state.meta?.model_data?.first_score_date||'';
      input.max=state.meta?.model_data?.last_score_date||'';
      input.value=research.signalDate;
      input.title=`模型信号日（有分数：${input.min} 至 ${input.max}）；与行情截面日期分开`;
      input.disabled=!available||Boolean(bootstrap);
    });
    root.querySelectorAll('[data-research-control="model"]').forEach(input=>{
      input.value=research.model;
      input.disabled=!available||Boolean(bootstrap);
    });
    research.featureDate=state.asOf||research.featureDate;
    const dateInput=root.querySelector('[data-control="date"]');
    if(dateInput){dateInput.min=state.meta?.first_date||'';dateInput.max=state.meta?.last_date||'';dateInput.value=state.asOf;dateInput.disabled=Boolean(bootstrap)||!['market','features'].includes(state.pageName);}
    renderFeatureIntro();
    renderRecentResearch();
  }
  function featureStockName(){return research.stockName||(state.stock?.ts_code===research.stockCode?state.stock.name:research.stockCode);}
  function renderFeatureIntro(){
    const copy={
      observations:research.features?`${featureStockName()}（${research.stockCode}）的六项观察值按 ${research.featureDate||'—'} 行情截面计算。它们便于阅读个股和主板分布；模型输入的完整目录见“模型特征”。模型评分归因另按所选信号日读取。`:`正在读取 ${featureStockName()}（${research.stockCode}）的量价观察…`,
      catalog:'三套模型共用 69 项截面输入，其中 28 项为基础量价指标、41 项为扩展特征。分组计数和定义取自模型交付目录；日线序列另见独立视图。',
      sequence:research.sequence?`${featureStockName()}（${research.stockCode}）的实际模型窗口为 ${research.sequence.start_date} 至 ${research.sequence.as_of}，共 ${research.sequence.window} 个交易日、11 个通道。`:research.loadingStock?`正在读取 ${featureStockName()}（${research.stockCode}）的日线序列…`:`${featureStockName()}（${research.stockCode}）的日线序列采用近 60 个交易日、11 个量价与交易状态通道；前 9 项按每日沪深主板截面排名，缺失值按模型输入规则填 0。`
    };
    slot('feature-intro').textContent=copy[research.featureView]||copy.observations;
  }
  function renderRecentResearch(){
    const buttons=research.recent.length?research.recent.map(item=>`<button type="button" data-research-stock="${escape(item.code)}" data-stock-name="${escape(item.name)}" title="重新查看${escape(item.name)}">${escape(item.name)}<span>${escape(item.code)}</span></button>`).join(''):'';
    const feature=slot('research-recent'),market=slot('market-recent');
    if(feature)feature.innerHTML=buttons;
    if(market)market.innerHTML=buttons;
  }
  async function loadSequenceWindow(){
    if(state.pageName!=='features'||research.featureView!=='sequence')return;
    const code=research.stockCode,asOf=research.featureDate,cacheKey=`${code}|${asOf}`,request=++research.sequenceRequest;
    if(sequenceCache.has(cacheKey)){research.sequence=sequenceCache.get(cacheKey);renderSequence();renderFeatureIntro();return;}
    slot('sequence-count').textContent='';slot('sequence-summary').textContent='正在读取所选股票的模型输入…';
    slot('sequence-head').replaceChildren();slot('sequence-body').replaceChildren();
    try{
      const body=await read(`/api/stocks/${encodeURIComponent(code)}/sequence?as_of=${encodeURIComponent(asOf)}`);
      if(request!==research.sequenceRequest||state.pageName!=='features'||research.featureView!=='sequence')return;
      sequenceCache.set(cacheKey,body);research.sequence=body;renderSequence();renderFeatureIntro();
    }catch(error){
      if(request!==research.sequenceRequest)return;
      research.sequence=null;slot('sequence-count').textContent='';slot('sequence-summary').textContent=error.message;
      slot('sequence-body').replaceChildren();renderFeatureIntro();
    }
  }
  async function loadResearch(){
    const page=state.pageName,request=++research.request;
    if(page==='market')return;
    if(page==='agents'){syncDebateWorkbench();return;}
    try{
      const asOf=page==='features'?research.featureDate:(research.signalDate||state.meta?.model_data?.last_score_date||'');
      if(!asOf)throw new Error(state.meta?.model_data?.message||'模型评分日期暂不可用');
      const params=new URLSearchParams({as_of:asOf,model:research.model});
      if(page==='features'){
        params.set('score_as_of',research.signalDate||state.meta?.model_data?.last_score_date||'');
        params.set('code',research.stockCode);
        const key=[asOf,research.signalDate,research.stockCode,research.model].join('|');
        if(!featureCache.has(key)){
          slot('feature-title').textContent='正在读取量价特征…';
          status('正在读取个股特征与归因…');
          const pending=read('/api/modules/features?'+params).then(body=>{
            featureCache.set(key,body);
            return body;
          }).catch(error=>{
            featureCache.delete(key);
            throw error;
          });
          featureCache.set(key,pending);
        }
        if(request!==research.request)return;
        research.features=await featureCache.get(key);
        if(request!==research.request)return;
        renderFeatures();renderFeatureCatalog();renderFeatureIntro();
        if(research.featureView==='sequence')await loadSequenceWindow();
      }else{
        params.set('rule',research.rule);params.set('period',research.period);
        const key=[page,asOf,research.model,research.rule,research.period].join('|');
        if(!moduleCache.has(key))moduleCache.set(key,await read('/api/modules/'+page+'?'+params));
        if(request!==research.request)return;
        research.data=moduleCache.get(key);
        if(!research.data.candidates.some(s=>s.ts_code===research.agentCode)){
          research.agentCode=research.data.candidates.some(s=>s.ts_code===state.code)?state.code:research.data.candidates[0]?.ts_code;
        }
        if(page==='agents'&&research.agentCode){
          const selectedKey=key+'|'+research.agentCode;
          if(!moduleCache.has(selectedKey)){
            const selectedParams=new URLSearchParams(params);selectedParams.set('code',research.agentCode);
            moduleCache.set(selectedKey,await read('/api/modules/agents?'+selectedParams));
          }
          if(request!==research.request)return;
          research.data=moduleCache.get(selectedKey);
        }
        renderResearch();
      }
      status('');
    }catch(error){if(request===research.request){status(error.message);if(page==='features'){
      research.features=null;slot('feature-title').textContent='特征暂不可用';
      slot('feature-metrics').replaceChildren();slot('feature-formula').textContent=error.message;
      slot('attribution-method').textContent='';slot('attribution-summary').replaceChildren();slot('attribution-factors').replaceChildren();
      root.querySelectorAll('[data-research-chart="distribution"],[data-research-chart="feature-line"]').forEach(s=>s.replaceChildren());
      renderFeatureIntro();
    }else{
      if(page==='portfolio'){
        slot('portfolio-context').textContent='所选日期暂不可用';
        slot('portfolio-table').innerHTML='<tr><td colspan="8">无法读取候选，请选择有模型评分的交易日</td></tr>';
      }
      if(page==='agents')status(error.message);
    }}}
  }
  async function syncDebateWorkbench(){
    if(state.pageName!=='agents')return;
    const request=++debateSyncRequest;
    const model=research.model;
    const asOf=research.signalDate||state.meta?.model_data?.last_score_date||'';
    let run=null;
    try{
      const params=new URLSearchParams({as_of:asOf,model});
      const result=await read('/api/modules/agents?'+params);
      if(request!==debateSyncRequest)return;
      run=['completed','incomplete'].includes(result.agent_run?.status)?result.agent_run:null;
    }catch(error){
      if(request!==debateSyncRequest)return;
      status(error.message);
    }
    const query=new URLSearchParams({
      parent_origin:location.origin,
      model_source:model,
      as_of:asOf,
      tab:debateTab,
    });
    if(run?.hash_prefix)query.set('hash',run.hash_prefix);
    const url=`${debateBase}?${query}`;
    const frame=root.querySelector('[data-slot="debate-frame"]');
    if(!frame.getAttribute('src'))frame.src=url;
    if(debateReady){
      frame.contentWindow.postMessage({type:'dashboard-context',model_source:model,as_of:asOf,hash:run?.hash_prefix||'',tab:debateTab},debateOrigin);
      if(pendingChatPrefill){
        frame.contentWindow.postMessage({type:'dashboard-chat-prefill',text:pendingChatPrefill},debateOrigin);
        pendingChatPrefill='';
      }
    }
  }
  window.addEventListener('message',event=>{
    const frame=root.querySelector('[data-slot="debate-frame"]');
    if(event.source!==frame.contentWindow||event.origin!==debateOrigin)return;
    if(event.data?.type==='debate-ready'){
      debateReady=true;
      syncDebateWorkbench();
    }else if(event.data?.type==='debate-resize'&&Number.isFinite(event.data.height)){
      frame.style.height=Math.max(560,Math.min(12000,event.data.height))+'px';
    }else if(event.data?.type==='debate-tab'&&['stock','final','models','prompts'].includes(event.data.tab)){
      debateTab=event.data.tab;
    }
  });
  function renderResearch(){
    if(state.pageName==='features'&&research.features)renderFeatures();
    if(state.pageName==='features'&&research.features)renderFeatureCatalog();
    if(state.pageName==='features'&&research.sequence)renderSequence();
    if(state.pageName==='models'&&research.data)renderModels();
    if(state.pageName==='portfolio'&&research.data)renderPortfolio();
    if(state.pageName==='agents'&&research.data)renderAgents();
  }
  function insightEvidence(kind){
    const stock=state.stock?.quote||{},breadth=state.market?.breadth||{};
    if(kind==='market')return `市场日期 ${stock.trade_date||state.asOf}；${state.stock?.name||state.code}（${state.stock?.ts_code||state.code}）收盘 ${fmt(stock.close)} 元，涨跌 ${sign(stock.pct_chg)}%；${state.stock?.bars?.length||0} 根所选周期行情。全市场上涨 ${fmt(breadth.rising,0)} 只、下跌 ${fmt(breadth.falling,0)} 只、平盘 ${fmt(breadth.unchanged,0)} 只，成交额 ${fmt(breadth.amount_yi,0)} 亿元。请解释涨跌概览和该股当前可见走势，注明统计口径，不推测未来涨幅。`;
    if(kind==='features'&&research.features){
      const data=research.features,f=data.features.find(item=>item.key===research.key)||data.features[0],a=data.model_attribution;
      return `行情特征日期 ${data.as_of}；模型评分日 ${data.score_as_of}。股票 ${featureStockName()}，观察项 ${f.label}=${fmt(f.value)}${f.unit}，截面分位 ${fmt(f.percentile,1)}，样本 ${f.valid} 只。模型 ${modelDisplayName(a?.model_id||research.model,a?.model_name)}，${a?.available?`评分名次 ${a.rank}/${a.pool_size}，已展示主要特征贡献`:`无可用单股归因：${a?.reason||'无记录'}`}。请解释观察项和模型归因的区别，不把分位或贡献解释成涨幅概率。`;
    }
    if(kind==='feature-catalog'&&research.features)return `${slot('catalog-description').textContent} 当前筛选为${research.catalogGroup==='all'?'全部分组':research.catalogGroup}，${slot('catalog-count').textContent}。请结合具体定义和设计依据解释这组输入的作用，并区分截面特征与 LSTM 序列通道。`;
    if(kind==='models'&&research.data)return `${slot('model-insight').textContent} 请依据当前策略、区间与曲线解释模型表现差异，并说明统计检验限制。`;
    if(kind==='portfolio'&&research.data)return `${slot('portfolio-insight').textContent} 请依据当前持仓计划解释排名、现金预留、行业上限与整手预算如何影响配置；不要把收盘估算称为实际成交。`;
    return '';
  }
  function metric(label,value,detail='',tone='accent'){
    return `<div class="sd-metric-${tone}"><label>${label}</label><strong>${value}</strong>${detail?`<span>${detail}</span>`:''}</div>`;
  }
  const valueTone=value=>Number(value)>0?'positive':Number(value)<0?'negative':'neutral';
  function chartAxes(svg,yMin,yMax,unit,ticks=[]){
    const {width,height}=dimensions(svg),L=55,R=16,T=20,B=height-30;
    const x0=L,x1=width-R,y=v=>T+(yMax-v)/(yMax-yMin||1)*(B-T);
    for(let i=0;i<4;i++){
      const v=yMin+(yMax-yMin)*i/3,yy=y(v);
      shape(svg,'line',{x1:L,x2:x1,y1:yy,y2:yy,stroke:'var(--sd-line)','stroke-width':1});
      shape(svg,'text',{x:L-8,y:yy+4,'text-anchor':'end'},fmt(v,Math.abs(v)>100?0:1));
    }
    shape(svg,'text',{x:L,y:12},unit);
    for(const tick of ticks)shape(svg,'text',{x:L+tick.p*(x1-L),y:height-8,'text-anchor':tick.p===0?'start':tick.p===1?'end':'middle'},tick.label);
    return {width,height,x0,x1,T,B,y};
  }
  function lines(svg,series,unit,palette=colors){
    const values=series.flatMap(s=>s.points.map(p=>p.value)).filter(Number.isFinite);
    if(!values.length){dimensions(svg);return;}
    const dates=series.flatMap(s=>s.points.map(p=>Date.parse(p.date))),first=Math.min(...dates),last=Math.max(...dates);
    const lo=Math.min(...values),hi=Math.max(...values),padding=(hi-lo||1)*.1;
    const ticks=[first,(first+last)/2,last].map((v,i)=>({p:i/2,label:new Date(v).toISOString().slice(0,7)}));
    const axes=chartAxes(svg,unit==='净值'?Math.max(0,lo-padding):lo-padding,hi+padding,unit,ticks),x=d=>axes.x0+(Date.parse(d)-first)/(last-first||1)*(axes.x1-axes.x0);
    for(let i=0;i<series.length;i++){
      const s=series[i],points=s.points.map(p=>`${x(p.date)},${axes.y(p.value)}`).join(' ');
      const stroke=palette[i%palette.length];
      shape(svg,'polyline',{points,fill:'none',stroke,'stroke-width':2,
        'stroke-dasharray':s.kind==='model'||s.kind==='benchmark'?modelChartDashes[i%modelChartDashes.length]:'none','pointer-events':'none'});
      s.points.forEach((p,j)=>{
        const mark=shape(svg,'circle',{cx:x(p.date),cy:axes.y(p.value),r:j===s.points.length-1?3:2,
          fill:stroke,opacity:j===s.points.length-1?1:.5,'data-tooltip':`${s.label} ${p.date} ${fmt(p.value)}${unit}`});
        shape(mark,'title',{},`${s.label} ${p.date} ${fmt(p.value)}${unit}`);
      });
    }
  }
  function renderFeatures(){
    const data=research.features,f=data.features.find(f=>f.key===research.key)||data.features[0];
    slot('feature-menu').innerHTML=data.features.map(f=>`<button type="button" data-feature="${f.key}" aria-pressed="${research.key===f.key}">${f.label}<span>${f.group}</span></button>`).join('');
    slot('feature-title').textContent=f.label;
    slot('feature-date').textContent=`行情截面 ${data.as_of} · ${featureStockName()}`;
    slot('feature-metrics').innerHTML=metric('个股数值',fmt(f.value)+f.unit,data.missing_reason||'',valueTone(f.value))+
      metric('截面分位',fmt(f.percentile,1)+'%',`有效样本 ${fmt(f.valid,0)} 只`)+metric('截面中位数',fmt(f.median)+f.unit);
    slot('feature-formula').textContent=`${f.formula}。统计范围：${data.population}，${fmt(data.population_count,0)} 只。`;
    const svg=root.querySelector('[data-research-chart="distribution"]');
    const max=Math.max(...f.distribution.map(b=>b.count),1),axes=chartAxes(svg,0,max*1.1,'只');
    const lo=f.distribution[0].lo,hi=f.distribution.at(-1).hi,bw=(axes.x1-axes.x0)/f.distribution.length;
    f.distribution.forEach((b,i)=>{
      const mark=shape(svg,'rect',{x:axes.x0+i*bw+1,y:axes.y(b.count),width:Math.max(1,bw-3),height:axes.B-axes.y(b.count),
        fill:(b.lo+b.hi)/2<0?'var(--sd-green)':'var(--sd-red)',opacity:.72,'data-tooltip':`${fmt(b.lo)}–${fmt(b.hi)}${f.unit}：${b.count}只`});
      shape(mark,'title',{},mark.getAttribute('data-tooltip'));
    });
    [lo,(lo+hi)/2,hi].forEach((v,i)=>shape(svg,'text',{x:axes.x0+i/2*(axes.x1-axes.x0),y:axes.height-8,'text-anchor':i===0?'start':i===2?'end':'middle'},fmt(v,1)+f.unit));
    if(f.value!=null){
      const xx=axes.x0+(Math.max(lo,Math.min(hi,f.value))-lo)/(hi-lo)*(axes.x1-axes.x0);
      shape(svg,'line',{x1:xx,x2:xx,y1:axes.T,y2:axes.B,stroke:'var(--sd-red)','stroke-width':2});
      shape(svg,'text',{x:Math.min(axes.x1-35,Math.max(axes.x0+8,xx+5)),y:axes.T+12},'个股');
    }
    lines(root.querySelector('[data-research-chart="feature-line"]'),[{label:f.label,points:f.timeseries}],f.unit);
    const direction=f.value==null?'当前股票在该截面没有有效值。':
      `当前值 ${fmt(f.value)}${f.unit}，位于当日有效样本第 ${fmt(f.percentile,1)} 百分位（样本中位数 ${fmt(f.median)}${f.unit}）。`;
    slot('feature-insight').textContent=`${featureStockName()} 的${f.label}：${direction}分布图与趋势图均按 ${data.as_of} 行情计算。${data.model_attribution?.available?`${modelDisplayName(data.model_attribution.model_id,data.model_attribution.model_name)} 在 ${data.score_as_of} 信号日排名第 ${data.model_attribution.rank}/${data.model_attribution.pool_size}；归因贡献表示对模型分数的影响。`:`${data.score_as_of} 信号日无可用个股归因：${data.model_attribution?.reason||'未提供对应记录'}。`}`;
    const a=data.model_attribution;
    if(!a?.available){
      slot('attribution-method').textContent='';
      slot('attribution-caption').textContent='归因只展示已有的单股解释记录；无记录时不以模型分数方向替代解释。';
      slot('attribution-summary').innerHTML=`<p>${escape(a?.reason||'该日期没有已交付的个股归因')}</p>`;
      slot('attribution-factors').replaceChildren();
      return;
    }
    slot('attribution-method').textContent=modelDisplayName(a.model_id,a.model_name)+' · '+a.method+' · 信号日 '+data.score_as_of;
    slot('attribution-caption').textContent=a.model_id==='lstm_SF69'?'LSTM 的解释同时涉及 69 项截面特征与 60×11 序列通道；本表只列影响绝对值最大的 8 项。':'本表列出影响绝对值最大的 8 项；正贡献抬高模型分数，负贡献压低模型分数。数值不是收益概率。';
    slot('attribution-summary').innerHTML=metric('模型名次',`第 ${fmt(a.rank,0)} 名`,`全池 ${fmt(a.pool_size,0)} 只`)+
      metric('全池百分位',fmt(a.percentile,2)+'%','用于不同模型间比较');
    const factorMax=Math.max(...a.factors.map(row=>Math.abs(Number(row['贡献'])||0)),1e-9);
    slot('attribution-factors').innerHTML=a.factors.map(row=>{
      const impact=Number(row['贡献'])||0,width=Math.max(2,Math.abs(impact)/factorMax*100);
      return `<div class="sd-factor-row"><span>${escape(row['中文名'])}</span><span>${fmt(row['数值'],4)}</span><div class="sd-factor-track"><i class="${impact>=0?'sd-factor-positive':'sd-factor-negative'}" style="width:${width}%"></i></div><strong class="${impact>=0?'sd-factor-positive-text':'sd-factor-negative-text'}">${impact>=0?'+':''}${fmt(impact,5)}</strong></div>`;
    }).join('');
  }
  function renderFeatureCatalog(){
    const rows=research.features?.model_feature_catalog||[];
    const matched=rows.filter(row=>row['来源']!=='LSTM 序列通道（1.6）');
    const search=research.catalogQuery.trim().toLocaleLowerCase();
    const groupLabel=value=>String(value||'').replace(/^[A-H]\s+/,'').replace('仓库原特征','基础量价');
    const groups=[...new Set(matched.map(row=>groupLabel(row['分组'])).filter(Boolean))];
    if(!groups.includes(research.catalogGroup))research.catalogGroup='all';
    const filtered=matched.filter(row=>(research.catalogGroup==='all'||groupLabel(row['分组'])===research.catalogGroup)&&
      (!search||[row['中文名'],row['计算'],row['理由'],groupLabel(row['分组'])].some(value=>String(value||'').toLocaleLowerCase().includes(search))));
    slot('catalog-title').textContent='模型特征';
    slot('catalog-count').textContent=`${filtered.length} / ${matched.length} 项`;
    slot('catalog-description').textContent='XGBoost、随机森林与 LSTM 共用 69 项截面输入；28 项基础量价特征与 41 项扩展特征按分组列出。每项保留中文名称、计算定义和设计依据。';
    const groupControl=root.querySelector('[data-feature-group]');
    groupControl.innerHTML='<option value="all">全部分组</option>'+groups.map(group=>`<option value="${escape(group)}">${escape(group)}</option>`).join('');
    groupControl.value=groups.includes(research.catalogGroup)?research.catalogGroup:'all';
    slot('feature-catalog').innerHTML=filtered.map(row=>`<tr><td><strong>${escape(row['中文名'])}</strong></td><td>${escape(groupLabel(row['分组']))}</td><td>${escape(row['计算'])}</td><td>${escape(row['理由']||'—')}</td></tr>`).join('')||'<tr><td colspan="4">没有匹配的特征</td></tr>';
    const f69Count=rows.filter(row=>row['来源']!=='LSTM 序列通道（1.6）').length;
    const original=rows.filter(row=>row['来源']==='原特征（仓库 F28）').length,expanded=f69Count-original;
    slot('feature-stats').innerHTML=metric('共同截面输入',`${f69Count} 项`,'三套模型共用')+metric('基础量价特征',`${original} 项`,'已有输入')+metric('扩展特征',`${expanded} 项`,'新增量价信息')+metric('LSTM 日线通道',`${rows.length-f69Count} 项`,'另含 60 日序列');
    const groupCounts=groups.map(group=>({group,count:matched.filter(row=>groupLabel(row['分组'])===group).length}));
    const groupMax=Math.max(1,...groupCounts.map(item=>item.count));
    slot('feature-groups').innerHTML=groupCounts.map(item=>`<div class="sd-feature-group-row"><span>${escape(item.group)}</span><i><b style="width:${item.count/groupMax*100}%"></b></i><strong>${item.count}</strong></div>`).join('');
    slot('catalog-insight').textContent=`目录显示 ${original} 项基础特征与 ${expanded} 项扩展特征。上方六项用于个股和截面观察，模型仍读取完整的 ${f69Count} 项输入；LSTM 另接收 60 个交易日的序列。`;
  }
  function renderSequence(){
    const data=research.sequence;if(!data)return;
    const definitions=(research.features?.model_feature_catalog||[]).filter(row=>row['来源']==='LSTM 序列通道（1.6）');
    const labels=definitions.map(row=>row['中文名']);
    const channelNames=definitions.map(row=>row['名称']);
    slot('sequence-count').textContent=`${data.window} 日 · ${channelNames.length} 项`;
    slot('sequence-description').textContent=`${featureStockName()}（${research.stockCode}）的序列输入：前 9 项每天按沪深主板横截面排名后映射至（−0.5，0.5）；涨跌停保留 −0.5、0、0.5 编码，交易状态以 0/1 表示。股票当日无记录时按模型规则填 0。`;
    slot('sequence-summary').innerHTML=metric('起始交易日',data.start_date)+metric('结束交易日',data.as_of)+metric('输入形状',`${data.window} × ${channelNames.length}`,'交易日 × 通道');
    slot('sequence-head').innerHTML=`<tr><th scope="col">交易日</th>${labels.map(label=>`<th scope="col" title="${escape(label)}">${escape(label)}</th>`).join('')}</tr>`;
    slot('sequence-body').innerHTML=data.rows.map(row=>`<tr><th scope="row">${escape(row.date)}</th>${channelNames.map((name,index)=>{
      const value=Number(row[name])||0,kind=value>0?'positive':value<0?'negative':'neutral';
      return `<td class="sd-sequence-${kind}" title="${escape(row.date)} · ${escape(labels[index])}：${fmt(value,3)}"><span>${fmt(value,2)}</span></td>`;
    }).join('')}</tr>`).join('');
    slot('sequence-definitions').innerHTML=definitions.map(row=>`<tr><td>${escape(row['中文名'])}</td><td>${escape(row['计算'])}</td></tr>`).join('');
  }
  function renderModels(){
    const data=research.data;
    slot('model-provenance').textContent='统一策略下的收益与风险';
    root.querySelectorAll('[data-model-period]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.modelPeriod===research.period)));
    const summaries=data.summaries.map(row=>({...row,label:modelDisplayName(row.model_id,row.label)}));
    const entries=[...data.series.map(item=>({...item,label:modelDisplayName(item.model_id,item.label),kind:'model'})),...data.benchmarks.map(item=>({...item,kind:'benchmark'}))];
    slot('model-legend').innerHTML=entries.map((item,i)=>`<span><svg width="22" height="8" aria-hidden="true"><line x1="0" x2="22" y1="4" y2="4" stroke="${modelChartColors[i%modelChartColors.length]}" stroke-width="2.5" stroke-dasharray="${modelChartDashes[i%modelChartDashes.length]}"/></svg>${escape(item.label)}</span>`).join('');
    slot('model-period').textContent=`${data.period_label} · ${data.nav_last_date} 净值 · 信号分数截至 ${data.score_last_date}`;
    lines(root.querySelector('[data-research-chart="models"]'),entries,'净值',modelChartColors);
    const bestExcess=summaries.reduce((a,b)=>a.excess_pct>b.excess_pct?a:b);
    const shallowDrawdown=summaries.reduce((a,b)=>a.max_drawdown_pct>b.max_drawdown_pct?a:b);
    const meanIc=summaries.reduce((sum,row)=>sum+row.rank_ic,0)/summaries.length;
    slot('model-metrics').innerHTML=metric('三模型平均 Rank IC',fmt(meanIc,4),'同一测试期',valueTone(meanIc))+
      metric('最高年化超额',fmt(bestExcess.excess_pct)+'%',bestExcess.label,valueTone(bestExcess.excess_pct))+
      metric('最大回撤最小',fmt(shallowDrawdown.max_drawdown_pct)+'%',shallowDrawdown.label,valueTone(shallowDrawdown.max_drawdown_pct))+
      metric('训练时间范围',`${fmt(Math.min(...summaries.map(row=>row.training_seconds)),1)}–${fmt(Math.max(...summaries.map(row=>row.training_seconds)),1)}秒`,'模型最终阶段记录','neutral');
    const directionClass=value=>Number(value)>0?'sd-up':Number(value)<0?'sd-down':'';
    slot('model-table').innerHTML=summaries.map((row,i)=>`<tr><td><span class="sd-model-name"><i style="background:${colors[i]}"></i>${escape(row.label)}</span></td><td class="${directionClass(row.rank_ic)}">${fmt(row.rank_ic,4)}</td><td class="${directionClass(row.annual_pct)}">${fmt(row.annual_pct)}%</td><td class="${directionClass(row.excess_pct)}">${fmt(row.excess_pct)}%</td><td class="${directionClass(row.max_drawdown_pct)}">${fmt(row.max_drawdown_pct)}%</td><td>${fmt(row.turnover,1)}倍</td><td>${fmt(row.annual_cost_pct)}%</td><td>${fmt(row.training_seconds,1)}秒</td></tr>`).join('');
    slot('strategy-title').textContent=data.rule_label;
    slot('strategy-description').textContent=data.rule_description;
    slot('strategy-facts').innerHTML=`<dt>策略本金</dt><dd>${fmt(data.summaries[0].capital,0)}元</dd><dt>模型数量</dt><dd>3 套</dd><dt>净值截止</dt><dd>${data.nav_last_date}</dd><dt>评分截止</dt><dd>${data.score_last_date}</dd>`;
    slot('model-discipline').textContent=data.discipline;
    const ranked=[...summaries].sort((a,b)=>b.annual_pct-a.annual_pct);
    slot('model-insight').textContent=`当前比较 ${data.rule_label}，本金 ${fmt(data.summaries[0].capital,0)} 元，${data.period_label}净值截至 ${data.nav_last_date}。年化收益排序为 ${ranked.map(row=>`${row.label} ${fmt(row.annual_pct)}%`).join('、')}；最高年化超额为 ${bestExcess.label} ${fmt(bestExcess.excess_pct)}%，回撤较小的是 ${shallowDrawdown.label} ${fmt(shallowDrawdown.max_drawdown_pct)}%。上述数据来自已交付回测，模型间 Rank IC 与组合超额差异${String(data.discipline).includes('未达到统计显著')?'未达到统计显著':''}。`;
  }
  function calculatePortfolio(){
    const budget=research.budget,reserve=budget*research.reserve/100,available=budget-reserve,slotBudget=available/research.count;
    let spent=0,fees=0;const holdings=[],blocked=[],industry={};
    for(const s of research.data.candidates){
      if(holdings.length>=research.count)break;
      if((industry[s.industry||'未分类']||0)>=research.industryCap){blocked.push({...s,reason:'行业数量上限'});continue;}
      const feeRate=research.fee/10000;
      let lots=Math.floor(slotBudget/(s.close*100*(1+feeRate)));
      while(lots>0&&lots*s.close*100+Math.max(5,lots*s.close*100*feeRate)>Math.min(slotBudget,available-spent-fees))lots--;
      if(!lots){blocked.push({...s,reason:'单个名额不足一手'});continue;}
      const amount=lots*100*s.close,cost=Math.max(5,amount*feeRate);
      holdings.push({...s,shares:lots*100,amount,cost,weight:budget?amount/budget*100:0});
      spent+=amount;fees+=cost;industry[s.industry||'未分类']=(industry[s.industry||'未分类']||0)+1;
    }
    research.holdings=holdings;research.blocked=blocked;
    return {holdings,blocked,spent,fees,cash:budget-spent-fees,reserve,budget};
  }
  function renderPortfolio(){
    const p=calculatePortfolio();
    slot('portfolio-metrics').innerHTML=metric('计划买入',fmt(p.spent,0)+'元',`${p.holdings.length} 只股票`)+
      metric('剩余现金',fmt(p.cash,0)+'元',`含预留 ${fmt(p.reserve,0)} 元`)+metric('预计买入费用',fmt(p.fees)+'元',`${research.fee} bps / 单笔最低5元`);
    const selectedModelName=modelDisplayName(research.data.model_id||research.model,research.data.model_label);
    slot('portfolio-context').textContent=research.data.as_of+' 收盘参考 · '+selectedModelName+' 模型前20名';
    slot('portfolio-table').innerHTML=p.holdings.length?p.holdings.map(s=>`<tr><td><button type="button" data-evidence-stock="${s.ts_code}" data-evidence-date="${research.data.as_of}">${escape(s.name)}<small>${s.ts_code} · ${escape(s.industry)}</small></button></td><td>${fmt(s.close)}</td><td>第 ${fmt(s.rank,0)} 名</td><td>${fmt(s.pct,2)}%</td><td>${fmt(s.shares,0)}</td><td>${fmt(s.amount,0)}</td><td>${fmt(s.weight,1)}%</td><td>候选前20名；行业未超限，整手可负担</td></tr>`).join(''):'<tr><td colspan="8">预算不足或约束下无可配置股票</td></tr>';
    const exposures={};p.holdings.forEach(s=>exposures[s.industry||'未分类']=(exposures[s.industry||'未分类']||0)+s.weight);
    slot('portfolio-exposure').innerHTML=Object.entries(exposures).map(([name,v])=>`<span>${escape(name)}<i style="width:${Math.max(3,v*1.5)}px"></i>${fmt(v,1)}%</span>`).join('');
    const reasons=p.blocked.map(item=>`<tr><td>第 ${fmt(item.rank,0)} 名 ${escape(item.name)}</td><td>${escape(item.reason)}</td><td>${escape(item.industry||'未分类')}</td></tr>`).join('');
    slot('portfolio-blocked').innerHTML=reasons?`<div class="sd-table-wrap"><table class="sd-research-table"><thead><tr><th>候选</th><th>未纳入原因</th><th>行业</th></tr></thead><tbody>${reasons}</tbody></table></div>`:'<p>本轮候选中没有因行业或单手预算被排除的股票。</p>';
    const reasonsText=p.blocked.length?`${p.blocked.length} 只排名靠前的候选因${[...new Set(p.blocked.map(item=>item.reason))].join('、')}未纳入。`:'本轮前20名中未有候选因行业上限或单手预算被跳过。';
    slot('portfolio-insight').textContent=`候选为 ${selectedModelName} 在 ${research.data.as_of} 信号日的全池排名前20名，依次检查，行业每类最多 ${research.industryCap} 只。将 ${fmt(p.budget,0)} 元预算先留存 ${research.reserve}%（${fmt(p.reserve,0)} 元），余款按 ${research.count} 个目标名额均分；按收盘参考价以 100 股整手买入，并估算单边 ${research.fee} bps 佣金（每笔最低 5 元）。${reasonsText}最终配置 ${p.holdings.length} 只，投入 ${fmt(p.spent,0)} 元，含费用剩余 ${fmt(p.cash,0)} 元。股数、入选项和现金随参数即时重算。`;
  }
  function renderAgents(){
    const selected=research.data.candidates.find(s=>s.ts_code===research.agentCode),evidence=research.data.model_evidence;
    if(!selected){
      slot('agent-stock').textContent='暂无候选';
      slot('agent-asof').textContent='候选日期 '+research.data.as_of;
      slot('workflow').replaceChildren();slot('agent-candidates').replaceChildren();slot('agent-content').replaceChildren();
      slot('agent-notice').textContent='该信号日没有可读取的模型候选。';
      return;
    }
    const asOf=research.data.as_of;
    slot('agent-asof').textContent='候选日期 '+asOf;
    const run=research.data.agent_run||{};
    slot('agent-status').textContent=run.available?`已辩 ${run.debated_count}/${run.submitted_count} · 终选 ${run.final_count}`:
      run.status==='incomplete'?`运行中 ${run.debated_count}/${run.submitted_count}`:'暂无匹配记录';
    slot('agent-notice').textContent=run.message||'当前仅显示已交付模型的候选和评分证据。';
    slot('workflow').innerHTML=research.data.workflow.map((step,i)=>`<div class="sd-workflow-step"><strong>${i+1}. ${escape(step.label)}</strong><span>${escape(step.detail)}</span></div>`).join('');
    slot('agent-candidates').innerHTML=research.data.candidates.map(s=>{
      const peerText=MODEL_PEERS.map(id=>s.peer_models[id]?`${MODEL_NAMES[id]} ${s.peer_models[id].rank}`:'').filter(Boolean).join(' · ');
      return `<button type="button" data-agent-stock="${s.ts_code}" aria-pressed="${s.ts_code===research.agentCode}"><strong>${fmt(s.rank,0)}　${escape(s.name)}</strong><span>${fmt(s.pct,2)}% · ${escape(peerText)}</span></button>`;
    }).join('');
    slot('agent-stock').textContent=selected.name+' · '+selected.ts_code;
    const peerRows=MODEL_PEERS.map((id,i)=>{
      const peer=selected.peer_models[id];
      return peer?`<tr><td><i class="sd-model-dot" style="background:${colors[i]}"></i>${MODEL_NAMES[id]}</td><td>第 ${fmt(peer.rank,0)} 名</td><td>${fmt(peer.pct,2)}%</td></tr>`:'';
    }).join('');
    const factors=evidence?.available?evidence.factors.map(row=>`<tr><td>${escape(row['中文名'])}</td><td>${fmt(row['数值'],4)}</td><td class="${Number(row['贡献'])>=0?'sd-factor-positive-text':'sd-factor-negative-text'}">${Number(row['贡献'])>=0?'+':''}${fmt(row['贡献'],5)}</td></tr>`).join(''):`<tr><td colspan="3">${escape(evidence?.reason||'该日期没有个股归因')}</td></tr>`;
    const finalRows=(run.final_picks||[]).map(item=>`<tr><td>${fmt(item.rank,0)}</td><td>${escape(item.name)}<small>${escape(item.ts_code)} · 模型第 ${fmt(item.model_rank,0)} 名</small></td><td>${escape(item.verdict||'')}</td><td>${escape(item.reason||'')}</td></tr>`).join('');
    const dialogue=run.dialogue||{},analysts=dialogue.analysts||[],argumentsList=dialogue.debate||[],verdict=dialogue.verdict;
    const analystHtml=analysts.map(item=>`<article><h4>${escape(item.role)} · ${escape(item.stance)}</h4><p>${escape(item.conclusion)}</p><p><strong>风险：</strong>${escape((item.risks||[]).join('；')||'未列出')}</p></article>`).join('');
    const argumentHtml=argumentsList.map(item=>`<article><h4>${escape(item.round)}</h4><p>${escape(item.argument)}</p></article>`).join('');
    const verdictHtml=verdict?`<div class="sd-agent-verdict"><h4>风险裁决 · ${escape(verdict.verdict)} · ${fmt(verdict.score,1)}分</h4><p>${escape(verdict.thesis)}</p><p><strong>主要风险：</strong>${escape((verdict.risks||[]).join('；')||'未列出')}</p><p><strong>执行意见：</strong>${escape(verdict.action)}</p></div>`:'';
    const runPanel=run.available?`<section class="sd-agent-run"><h3>终选结果</h3><p>${fmt(run.debated_count,0)} 只完成分析 · ${fmt(run.failed_count||0,0)} 只失败 · ${fmt(run.final_count||0,0)} 只进入终选</p><div class="sd-table-wrap"><table class="sd-research-table"><thead><tr><th>顺位</th><th>股票</th><th>裁决</th><th>入选理由</th></tr></thead><tbody>${finalRows||'<tr><td colspan="4">记录中没有终选名单</td></tr>'}</tbody></table></div><details class="sd-agent-transcript"><summary>查看该股的分析结论与多空交锋</summary>${analystHtml?`<div class="sd-arguments">${analystHtml}</div>`:'<p>存档中没有该股的分析师结论。</p>'}${argumentHtml?`<div class="sd-agent-rounds">${argumentHtml}</div>`:'<p>存档中没有该股的交锋记录。</p>'}${verdictHtml||'<p>存档中没有该股的风险裁决。</p>'}</details></section>`:'';
    slot('agent-content').innerHTML=`<div class="sd-agent-snapshot">${metric('信号日收盘',fmt(selected.close)+'元',`${asOf} · ${sign(selected.pct_chg)}%`)}${metric('所选模型名次',`第 ${fmt(selected.rank,0)} 名`,`全池百分位 ${fmt(selected.pct,2)}%`)}${metric('成交额',fmt(selected.amount_yi)+'亿元',escape(selected.industry))}</div><div class="sd-agent-evidence"><h3>三模型同日排名</h3><table class="sd-research-table"><thead><tr><th>模型</th><th>全池名次</th><th>百分位</th></tr></thead><tbody>${peerRows}</tbody></table><h3>所选模型评分归因</h3><p>${evidence?.available?`${escape(modelDisplayName(evidence.model_id,evidence.model_name))} · ${escape(evidence.method)}。贡献表示对该模型分数的影响，不是上涨概率。`:'该日期没有可用归因记录。'}</p><table class="sd-research-table"><thead><tr><th>特征</th><th>特征值</th><th>分数贡献</th></tr></thead><tbody>${factors}</tbody></table><button type="button" data-evidence-stock="${selected.ts_code}" data-evidence-date="${asOf}">打开信号日行情</button>${runPanel}</div>`;
  }
  function researchDialog(title,content){
    slot('research-dialog-title').textContent=title;slot('research-dialog-content').innerHTML=content;
    root.querySelector('[data-dialog="research"]').showModal();
  }
  function report(){
    const p=calculatePortfolio();
    researchDialog('组合配置报告',`<article class="sd-report"><h3>候选与资金假设</h3><dl><dt>模型</dt><dd>${escape(modelDisplayName(research.data.model_id||research.model,research.data.model_label))}</dd><dt>信号日期</dt><dd>${research.data.as_of}</dd><dt>候选来源</dt><dd>${escape(research.data.candidate_method)}</dd><dt>预算与现金预留</dt><dd>${fmt(p.budget,0)}元 / ${research.reserve}%</dd></dl><h3>持仓计划</h3><div class="sd-table-wrap"><table class="sd-research-table"><thead><tr><th>股票</th><th>模型名次</th><th>股数</th><th>参考金额</th></tr></thead><tbody>${p.holdings.map(s=>`<tr><td>${escape(s.name)}</td><td>第 ${fmt(s.rank,0)} 名</td><td>${fmt(s.shares,0)}</td><td>${fmt(s.amount,0)}元</td></tr>`).join('')}</tbody></table></div><p>计划买入 ${fmt(p.spent,0)}元；预计买入费用 ${fmt(p.fees)}元；剩余现金 ${fmt(p.cash,0)}元。</p><h3>成交条件</h3><p>按信号日收盘价估算，股数按100股取整，佣金按单边${research.fee} bps且每笔最低5元；行业上限为每行业${research.industryCap}只。没有模拟下一交易日成交、卖出成本、滑点或持仓收益，也不会改变历史回测曲线。</p><div class="sd-report-actions"><button type="button" class="sd-primary" data-research="print">打印 / 保存 PDF</button></div></article>`);
  }
  root.addEventListener('click',event=>{
    const b=event.target.closest('button');if(!b)return;
    if(b.dataset.feature){research.key=b.dataset.feature;renderFeatures();}
    if(b.dataset.agentStock){research.agentCode=b.dataset.agentStock;loadResearch();}
    if(b.dataset.researchStock){slot('search-results').hidden=true;loadStock(b.dataset.researchStock,b.dataset.stockName);}
    if(b.dataset.evidenceStock){
      root.querySelector('[data-dialog="research"]').close();
      const target=b.dataset.evidenceDate||research.data?.as_of||state.asOf;
      state.asOf=target;root.querySelector('[data-control="date"]').value=target;
      state.page=1;showPage('market');loadMarket();loadStock(b.dataset.evidenceStock);
    }
    const action=b.dataset.research;
    if(action==='close')root.querySelector('[data-dialog="research"]').close();
    if(action==='feature-info'&&research.features){
      const rows=research.features.model_feature_catalog.map(row=>`<tr><td>${escape(row['中文名'])}</td><td>${escape(row['计算'])}</td><td>${escape(row['理由']||'')}</td></tr>`).join('');
      researchDialog('特征口径与目录',`<p>量价截面与个股走势按行情日期 ${research.features.as_of} 计算；模型归因读取 ${research.features.score_as_of} 的信号评分。统计范围：${escape(research.features.population)}。</p><p>三套模型共用 69 项截面输入；LSTM 另读取过去 60 个交易日的 11 条序列通道。归因表显示所选股票在对应信号日的已交付分数贡献。</p><details><summary>查看 ${research.features.model_feature_catalog.length} 项特征与序列通道定义</summary><div class="sd-table-wrap sd-catalog-table"><table class="sd-research-table"><thead><tr><th>特征</th><th>计算定义</th><th>设计理由</th></tr></thead><tbody>${rows}</tbody></table></div></details><details><summary>查看页面绘制的常用量价指标</summary><div class="sd-catalog">${research.features.catalog.map(f=>`<span>${escape(f)}</span>`).join('')}</div></details>`);
    }
    if(action==='model-info')researchDialog('回测结果口径',`<p>净值与绩效指标来自 ${escape(research.data.source)} 的已交付结果文件。模型页面展示三套已完成回测，策略与区间选择只切换相应的保存结果；图表对比三模型与基准指数。</p><p>测试 Rank IC 衡量每日模型排名与后续收益排名的相关性。测试期三模型的 Rank IC 与组合超额差异均未达到统计显著。2015—2026 长期曲线属于补充分析，其中 2015—2022 区间的规则是在查看相应历史后确定，不能作为独立样本外结果。</p><p>图中净值截至 ${research.data.nav_last_date}；已交付模型评分截至 ${research.data.score_last_date}。组合页预算只影响单日资金测算，不会重算历史净值。</p>`);
    if(action==='candidate-info')researchDialog('候选与成交假设',`<p>${escape(research.data.candidate_method)}</p><p>组合页按模型名次读取候选，并以 ${research.data.as_of} 收盘价计算整手股数。行业上限、现金预留、预算和目标持仓数会即时重算资金安排；交易费仅按设置的单边佣金估算。</p><p>计划未模拟下一交易日成交、卖出成本、滑点或持仓收益，不构成下单指令，也不改变模型回测结果。</p>`);
    if(action==='agent-info')researchDialog('辩论流程与记录',`<p>流程依次读取模型前20名、两位独立分析师、多空交锋、风险裁决与整批终选。页面只展示本地保存且候选快照与当前模型排名、信号日行情相符的运行结果。</p><p>终选名单和单股角色记录来自已冻结的辩论账本；没有匹配账本时只显示模型输入与评分证据，不补写或生成角色发言。</p>`);
    if(action==='report')report();if(action==='print')window.print();
    if(action)saveUI();
  });
  root.addEventListener('change',event=>{
    const c=event.target.dataset.researchControl,v=Number(event.target.value);
    if(c==='model'){
      research.model=event.target.value;research.data=null;research.features=null;syncResearchControls();loadResearch();saveUI();return;
    }
    if(c==='signal-date'){
      if(!event.target.value)return;
      research.signalDate=event.target.value;research.data=null;research.features=null;research.agentCode=null;
      syncResearchControls();loadResearch();saveUI();return;
    }
    if(c==='rule'){
      research.rule=event.target.value;research.data=null;loadResearch();saveUI();return;
    }
    const settings={budget:['budget',0,1e8],reserve:['reserve',0,100],fee:['fee',0,100],'portfolio-count':['count',1,10],
      'industry-cap':['industryCap',1,3]};
    if(settings[c]){const [key,lo,hi]=settings[c];research[key]=Math.max(lo,Math.min(hi,Number.isFinite(v)?v:lo));event.target.value=research[key];renderResearch();}
  });
  new ResizeObserver(()=>requestAnimationFrame(renderResearch)).observe(root);
  function showPage(page,persistState=true){
    if(page!==state.pageName){
      ++research.request;
      research.data=null;
      research.features=null;
    }
    state.pageName=page;
    const showStockTools=['market','features'].includes(page);
    const toolbar=root.querySelector('.sd-toolbar');
    if(showStockTools){
      root.querySelector(`[data-view="${page}"] .sd-module-head`).after(toolbar);
      toolbar.hidden=false;
    }else toolbar.hidden=true;
    root.querySelector('.sd-tools').hidden=!showStockTools;
    if(page!=='market')syncResearchControls();
    slot('search-results').hidden=true;
    root.querySelectorAll('[data-view]').forEach(el=>el.hidden=el.dataset.view!==page);
    root.querySelectorAll('[data-page]').forEach(b=>{
      if(b.dataset.page===page)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');
    });
    renderCharts();if(page!=='market')loadResearch();if(persistState)saveUI();
  }
  root.addEventListener('click',event=>{
    const button=event.target.closest('button');if(!button)return;
    if(button.dataset.page)showPage(button.dataset.page);
    if(button.dataset.featureView){
      research.featureView=button.dataset.featureView;
      root.querySelectorAll('[data-feature-view]').forEach(tab=>tab.setAttribute('aria-selected',String(tab===button)));
      root.querySelectorAll('[data-feature-panel]').forEach(panel=>panel.hidden=panel.dataset.featurePanel!==research.featureView);
      renderFeatureIntro();
      if(research.featureView==='catalog'&&research.features)renderFeatureCatalog();
      if(research.featureView==='sequence'){
        if(research.features)loadSequenceWindow();
        else loadResearch();
      }
    }
    if(button.dataset.insightAsk){
      pendingChatPrefill=insightEvidence(button.dataset.insightAsk);
      showPage('agents');
      if(debateReady){
        root.querySelector('[data-slot="debate-frame"]').contentWindow.postMessage({type:'dashboard-chat-prefill',text:pendingChatPrefill},debateOrigin);
        pendingChatPrefill='';
      }
    }
    if(button.dataset.stock){slot('search-results').hidden=true;loadStock(button.dataset.stock,button.dataset.stockName);}
    if(button.dataset.list){state.list=button.dataset.list;state.page=1;root.querySelectorAll('[data-list]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));loadMarket();}
    if(button.dataset.sort){if(state.sort===button.dataset.sort)state.desc=!state.desc;else {state.sort=button.dataset.sort;state.desc=true;}state.page=1;loadMarket();}
    if(button.dataset.period){state.period=button.dataset.period;root.querySelectorAll('[data-period]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));loadStock();}
    if(button.dataset.count){state.count=Number(button.dataset.count);root.querySelectorAll('[data-count]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));loadStock();}
    if(button.dataset.sub){state.sub=button.dataset.sub;root.querySelectorAll('[data-sub]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderCharts();saveUI();}
    if(button.dataset.modelPeriod){
      research.period=button.dataset.modelPeriod;research.data=null;
      root.querySelectorAll('[data-model-period]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.modelPeriod===research.period)));
      loadResearch();saveUI();
    }
    const action=button.dataset.action;
    if(action==='previous'){state.page--;loadMarket();}if(action==='next'){state.page++;loadMarket();}
    if(action==='source')sourceDialog();if(action==='close')root.querySelector('[data-dialog="source"]').close();
    if(button.closest('.sd-module-head .sd-segment')){button.parentElement.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));}
  });
  root.addEventListener('change',event=>{
    const control=event.target.dataset.control;
    if(control==='board'){state.board=event.target.value;state.page=1;loadMarket();}
    if(control==='adjustment'){state.adjustment=event.target.value;loadStock();}
    if(control==='date'){
      state.asOf=event.target.value;research.featureDate=state.asOf;state.page=1;
      research.features=null;research.sequence=null;state.stock=null;research.stockError=false;research.loadingStock=true;
      ++research.request;++research.sequenceRequest;
      renderStockLoading(research.stockName||state.code,state.code);
      clearFeatureDetails(`正在读取 ${research.stockName||state.code} 的量价特征…`);
      renderFeatureIntro();loadMarket();loadStock();
    }
  });
  root.addEventListener('input',event=>{
    if(event.target.matches('[data-feature-search]')){research.catalogQuery=event.target.value;renderFeatureCatalog();return;}
    if(event.target.dataset.control!=='search')return;
    clearTimeout(searchTimer);++searchRequest;const q=event.target.value;
    state.query=q;persist('quant-stock-query',state.query);
    if(!q.trim()){slot('search-results').hidden=true;return;}
    searchTimer=setTimeout(async()=>{
      const request=++searchRequest;
      try{
        const body=await read('/api/stocks/search?q='+encodeURIComponent(q));
        if(request!==searchRequest)return;
        slot('search-results').innerHTML=body.results.length?body.results.map(s=>`<button type="button" data-stock="${s.ts_code}" data-stock-name="${escape(s.name)}">${escape(s.name)}<span>${s.ts_code}　${escape(s.market)}</span></button>`).join(''):'<p>未找到股票</p>';
        slot('search-results').hidden=false;
      }catch(error){status(error.message);}
    },180);
  });
  root.addEventListener('change',event=>{
    if(event.target.matches('[data-feature-group]')){research.catalogGroup=event.target.value;renderFeatureCatalog();}
  });
  root.addEventListener('keydown',event=>{
    if(event.key==='Escape')slot('search-results').hidden=true;
    if(event.key==='Enter'&&event.target.dataset.control==='search'){const first=slot('search-results').querySelector('[data-stock]');if(first){first.click();event.preventDefault();}}
  });
  document.addEventListener('click',event=>{if(!event.target.closest('.sd-search'))slot('search-results').hidden=true;});
  new ResizeObserver(renderCharts).observe(root);
  function restoreWidget(widget){
    const saved=widget?.modelContent;
    if(!saved)return;
    if(['r20','dev_ma20','vol20','dd20','from_low60','amp20'].includes(saved.feature))research.key=saved.feature;
    if(MODEL_PEERS.includes(saved.model))research.model=saved.model;
    if(/^(?:19|20)\d{2}-\d{2}-\d{2}$/.test(saved.signalDate||''))research.signalDate=saved.signalDate;
    if(['DB-K30-b8','DB-K3-b16','FH-K6-h4'].includes(saved.rule))research.rule=saved.rule;
    if(['test','long'].includes(saved.modelPeriod))research.period=saved.modelPeriod;
    if(Number.isFinite(saved.budget)){
      research.budget=Math.max(0,Math.min(1e8,saved.budget));
      root.querySelector('[data-research-control="budget"]').value=research.budget;
    }
    if(['market','features','models','portfolio','agents'].includes(saved.page))showPage(saved.page,false);
  }
  window.addEventListener('openai:set_globals',event=>restoreWidget(event.detail?.globals?.widgetState));
  async function init(){
    try{
      state.meta=await read('/api/meta');state.asOf=state.meta.last_date;
      research.signalDate=state.meta.model_data?.last_score_date||'';
      const dateInput=root.querySelector('[data-control="date"]');dateInput.value=state.asOf;dateInput.min=state.meta.first_date;dateInput.max=state.meta.last_date;if(bootstrap)dateInput.disabled=true;
      research.featureDate=state.asOf;
      root.querySelector('[data-control="search"]').value=state.query;
      syncResearchControls();
      root.querySelectorAll('[data-count]').forEach(b=>{b.setAttribute('aria-pressed',String(Number(b.dataset.count)===state.count));if(bootstrap&&Number(b.dataset.count)>60)b.disabled=true;});
      slot('data-status').textContent='日行情截至 '+state.meta.last_date;
      await Promise.all([loadMarket(),loadStock()]);
      restoreWidget(window.openai?.widgetState);
    }catch(error){status(error.message);}
  }
  init();
})();
