import streamlit as st
import datetime
import traceback
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# 加载环境变量 (读取 .env 中的 API 密钥)
load_dotenv()

st.set_page_config(page_title="TradingAgents UI", page_icon="📈", layout="wide")

st.title("📈 TradingAgents")
st.markdown("基于多智能体的大模型金融交易研究分析系统")

# 定义分析师映射
ANALYSTS_MAP = {
    "Market Analyst": "market",
    "Social Media Analyst": "social",
    "News Analyst": "news",
    "Fundamentals Analyst": "fundamentals"
}

# 侧边栏：配置参数
with st.sidebar:
    st.header("⚙️ 配置参数")
    
    ticker = st.text_input("股票代码 (Ticker)", value="NVDA", help="例如: AAPL, TSLA, SPY")
    
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    analysis_date = st.text_input("分析日期", value=today_str, help="格式: YYYY-MM-DD")
    
    st.divider()

    # 分析师团队多选框
    selected_analysts_display = st.multiselect(
        "选择分析师团队 (Analysts Team)",
        options=list(ANALYSTS_MAP.keys()),
        default=list(ANALYSTS_MAP.keys()),
        help="去除不需要的分析师可以加快运行速度且减少 Token 消耗"
    )
    selected_analysts = [ANALYSTS_MAP[k] for k in selected_analysts_display]
    
    depth_level = st.selectbox(
        "分析深度 (Research Depth)", 
        [
            "1 (Shallow) - 快速研究，少量辩论", 
            "3 (Medium) - 中等深度研究与辩论", 
            "5 (Deep) - 全面研究，深度辩论"
        ], 
        index=1
    )
    
    st.divider()

    provider = st.selectbox(
        "选择 LLM 供应商", 
        ["aliyun", "openai", "google", "anthropic", "xai", "openrouter", "ollama"]
    )
    
    # 根据提供商预设一个默认模型供用户修改
    default_deep_model = "qwen3-max-2026-01-23" if provider == "aliyun" else "gpt-5.2"
    default_quick_model = "qwen3.5-plus" if provider == "aliyun" else "gpt-5-mini"
    
    deep_model = st.text_input("深度思考模型 (Deep Model)", value=default_deep_model)
    quick_model = st.text_input("快速思考模型 (Quick Model)", value=default_quick_model)
    
    start_button = st.button("🚀 开始分析", type="primary", use_container_width=True)

# 主显示区
if start_button:
    if not ticker or not analysis_date:
        st.error("请完善股票代码和分析日期。")
    elif not selected_analysts:
        st.error("⚠️ 请至少选择一个分析师团队！")
    else:
        # 重写系统配置
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = provider
        config["deep_think_llm"] = deep_model
        config["quick_think_llm"] = quick_model
        config["max_debate_rounds"] = int(depth_level.split(" ")[0])

        # 建立 Tab 布局，避免页面杂乱
        tab_report, tab_workflow, tab_data = st.tabs(["📊 最终研究报告", "🔄 实时智能体工作流", "🗃️ 原始状态数据"])
        
        with tab_workflow:
            st.info(f"正在对 **{ticker}** 进行多智能体调研，请耐心留意流式工作流...")
            status_box = st.empty()
            status_box.markdown("⏳ 初始化模型与 Agent 网络...")

            try:
                # 初始化图结构
                ta = TradingAgentsGraph(
                    selected_analysts=selected_analysts, 
                    debug=True, 
                    config=config
                )
                
                # 设置股票代码到实例中，防止写入日志时文件夹名为 None
                ta.ticker = ticker
                
                # 开始执行工作流
                init_agent_state = ta.propagator.create_initial_state(ticker, analysis_date)
                args = ta.propagator.get_graph_args()
                
                # 采用页面原生的滚动模式，不使用定高的小框框
                log_container = st.container()
                seen_message_ids = set()
                
                final_state = None
                with st.spinner('🎯 Agents 正在激烈讨论与收集数据中...'):
                    for chunk in ta.graph.stream(init_agent_state, **args):
                        messages = chunk.get("messages", [])
                        for msg in messages:
                            msg_id = getattr(msg, 'id', None) or id(msg)
                            if msg_id not in seen_message_ids:
                                seen_message_ids.add(msg_id)
                                
                                msg_type = type(msg).__name__
                                name_attr = getattr(msg, 'name', '') or 'Agent / System'
                                
                                # 优化体验：彻底屏蔽框架底层的无用占位符 “Continue”，避免长刷屏
                                if getattr(msg, 'content', '') == 'Continue':
                                    continue
                                
                                with log_container:
                                    if msg_type == "AIMessage":
                                        with st.chat_message("assistant", avatar="🧠"):
                                            st.markdown(f"**{name_attr} 的分析与思考**")
                                            if getattr(msg, 'content', ''):
                                                st.info(msg.content)
                                                
                                            # 工具调用放入折叠面板
                                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                                for call in msg.tool_calls:
                                                    with st.expander(f"🛠️ 计划调用工具数据: `{call['name']}`", expanded=False):
                                                        st.json(call['args'])
                                                        
                                    elif msg_type == "HumanMessage":
                                        with st.chat_message("user", avatar="👤"):
                                            st.markdown(f"**🗣️ 外部指令 / 上下文注入**")
                                            st.markdown(getattr(msg, 'content', ''))
                                            
                                    elif msg_type == "ToolMessage":
                                        with st.chat_message("tool", avatar="⚙️"):
                                            st.markdown(f"**✅ 工具 `{name_attr}` 抓取数据完成**")
                                            # 把杂乱庞大的工具返回全部折叠！保持流水线的绝对干净
                                            with st.expander("查看工具原始返回数据", expanded=False):
                                                content_str = str(getattr(msg, 'content', ''))
                                                if len(content_str) > 3000:
                                                    st.markdown(content_str[:3000] + "\n\n> *(数据超长，已为您自动截断显示)*")
                                                else:
                                                    st.markdown(content_str)
                                            
                                    else:
                                        with st.chat_message("system", avatar="💻"):
                                            st.markdown(f"**🖥️ {msg_type}**")
                                            st.markdown(str(getattr(msg, 'content', '')))
                                            
                        final_state = chunk
                
                # 结束流程后的最后工作：
                ta.curr_state = final_state
                ta._log_state(analysis_date, final_state)
                decision = ta.process_signal(final_state["final_trade_decision"])
                status_box.success("✅ 多智能体讨论工作流完成！请点击上方『📊 最终研究报告』标签查看。")
                
            except Exception as e:
                status_box.error(f"❌ 运行过程中发生错误：{str(e)}")
                st.code(traceback.format_exc(), language="python")
                decision = None
                
        # 分离结果到独立的 Tab，美观度大幅上升
        with tab_report:
            if final_state and decision:
                st.subheader(f"📊 {ticker} [{analysis_date}] 执行决策结果")
                st.markdown(decision)
                
                st.divider()
                st.markdown("### 📎 各大团队子研究报告归档")
                # 把每个专业分析师生成的报告也收纳起来
                reports_map = [
                    ("market_report", "📈 量化市场分析报告"),
                    ("sentiment_report", "🌐 社交媒体情绪报告"),
                    ("news_report", "🗞️ 宏观新闻与研报"),
                    ("fundamentals_report", "🏢 公司基本面分析报告")
                ]
                
                for r_key, r_name in reports_map:
                    content = final_state.get(r_key)
                    if content:
                        with st.expander(r_name, expanded=False):
                            st.markdown(content)
            else:
                st.info("👈 请在左侧配置参数并点击『开始分析』")
                
        # 原格式或 Debug 开发人员所需状态全收集
        with tab_data:
            if final_state:
                st.json(final_state)
            else:
                st.info("当前无运行时缓存数据。")
