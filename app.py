import streamlit as st
import datetime
import traceback
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# 加载环境变量 (读取 .env 中的 API 密钥)
load_dotenv()

st.set_page_config(page_title="TradingAgents UI", layout="wide")

st.title("📈 TradingAgents")
st.markdown("基于多智能体的大模型金融交易研究分析系统")

# 侧边栏：配置参数
with st.sidebar:
    st.header("⚙️ 配置参数")
    
    ticker = st.text_input("股票代码 (Ticker)", value="NVDA", help="例如: AAPL, TSLA, SPY")
    
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    analysis_date = st.text_input("分析日期", value=today_str, help="格式: YYYY-MM-DD")
    
    provider = st.selectbox(
        "选择 LLM 供应商", 
        ["aliyun", "openai", "google", "anthropic", "xai", "openrouter", "ollama"]
    )
    
    # 根据提供商预设一个默认模型供用户修改
    default_deep_model = "qwen3-max-2026-01-23" if provider == "aliyun" else "gpt-5.2"
    default_quick_model = "qwen3.5-plus" if provider == "aliyun" else "gpt-5-mini"
    
    deep_model = st.text_input("深度思考模型 (Deep Model)", value=default_deep_model)
    quick_model = st.text_input("快速思考模型 (Quick Model)", value=default_quick_model)
    
    depth_level = st.selectbox("分析深度", ["1 (Shallow)", "3 (Medium)", "5 (Deep)"], index=1)
    
    start_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

# 主显示区
if start_button:
    if not ticker or not analysis_date:
        st.error("请完善股票代码和分析日期。")
    else:
        # 重写系统配置
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = provider
        config["deep_think_llm"] = deep_model
        config["quick_think_llm"] = quick_model
        config["max_debate_rounds"] = int(depth_level.split(" ")[0])
        
        selected_analysts = ["market", "social", "news", "fundamentals"]

        st.info(f"正在对 **{ticker}** 进行多智能体调研，日期: {analysis_date}，请耐心等待...")
        
        # 建立一个占位符，用来在运行时展示状态
        status_box = st.empty()
        status_box.markdown("⏳ 初始化模型与 Agent 网络...")

        try:
            with st.spinner('Agents 正在激烈讨论与收集数据中...'):
                # 初始化图结构
                ta = TradingAgentsGraph(
                    selected_analysts=selected_analysts, 
                    debug=False, # 关闭终端的 debug trace，避免弄脏控制台
                    config=config
                )
                
                # 运行主流程
                # 由于 langgraph 在没有包装 generator 时会阻塞到完成，这里简化只展示最终结果
                final_state, decision = ta.propagate(ticker, analysis_date)
            
            status_box.success("✅ 分析完成！")
            
            st.divider()
            st.subheader(f"📊 {ticker} 最终交易决策与研究报告")
            
            # 渲染最终 Markdown 报告
            if decision:
                st.markdown(decision)
            else:
                st.warning("没有生成最终报告结果。")
                
            # 选项卡：额外展示一些后台状态或原始数据
            with st.expander("查看底层状态结构 (State Data)"):
                st.json(final_state)

        except Exception as e:
            st.error(f"❌ 运行过程中发生错误：{str(e)}")
            st.code(traceback.format_exc(), language="python")
