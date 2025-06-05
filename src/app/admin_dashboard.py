import streamlit as st
import os
from datetime import datetime, timedelta
from supabase import create_client
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

def run_dashboard():
    # Load environment variables
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    st.set_page_config(page_title="Admin Analytics Dashboard", layout="wide")
    st.title("📊 LegalDoc AI Admin Analytics Dashboard")

    # Connect to Supabase
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Get projects for filtering
    try:
        projects_data, _ = client.table("projects").select("id, name").execute()
        projects = projects_data[1] if projects_data and len(projects_data) > 1 else []
        project_options = {p["name"]: p["id"] for p in projects}
        project_options["All Projects"] = None
    except Exception as e:
        st.error(f"Error fetching projects: {e}")
        project_options = {"All Projects": None}

    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("End date", datetime.now())
    user_filter = st.sidebar.text_input("User ID (optional)")
    doc_filter = st.sidebar.text_input("Document ID (optional)")
    project_filter = st.sidebar.selectbox("Project", options=list(project_options.keys()))
    project_id_filter = project_options[project_filter]
    api_type_filter = st.sidebar.selectbox("API Type", ["All", "ocr", "embedding", "llm", "ner"])

    # Analytics tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["API Usage", "Project Analytics", "Document Statistics", "Cost Analysis", "Entities", "Relationships"])

    # Build query for API usage
    with tab1:
        query = client.table("api_usage_logs").select("*")
        query = query.gte("created_at", str(start_date)).lte("created_at", str(end_date + timedelta(days=1)))
        if user_filter:
            query = query.eq("user_id", user_filter)
        if doc_filter:
            query = query.eq("document_id", doc_filter)
        if api_type_filter != "All":
            query = query.eq("api_type", api_type_filter)
        if project_id_filter:
            try:
                project_docs_data, _ = client.table("documents").select("id").eq("project_id", project_id_filter).execute()
                project_doc_ids = [doc["id"] for doc in project_docs_data[1]] if project_docs_data and len(project_docs_data) > 1 else []
                if project_doc_ids:
                    query = query.in_("document_id", project_doc_ids)
                else:
                    st.warning(f"No documents found in the selected project.")
            except Exception as e:
                st.error(f"Error filtering by project: {e}")
        try:
            data, count = query.execute()
            logs = data[1] if data and len(data) > 1 else []
            df = pd.DataFrame(logs)
        except Exception as e:
            st.error(f"Failed to fetch analytics data: {e}")
            df = pd.DataFrame([])
        if not df.empty:
            st.subheader("Global Usage & Cost Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total API Calls", len(df))
            with col2:
                st.metric("Total Tokens Used", int(df["tokens_used"].sum()))
            with col3:
                st.metric("Total Cost (USD)", f"${df['cost_usd'].sum():.4f}")
            with col4:
                avg_cost_per_1k = (df['cost_usd'].sum() / df['tokens_used'].sum() * 1000) if df['tokens_used'].sum() > 0 else 0
                st.metric("Avg Cost per 1K Tokens", f"${avg_cost_per_1k:.4f}")
            st.subheader("API Usage Over Time")
            df['created_at'] = pd.to_datetime(df['created_at'])
            daily_usage = df.groupby([df['created_at'].dt.date, 'api_type']).size().reset_index(name='calls')
            daily_usage_pivot = daily_usage.pivot(index='created_at', columns='api_type', values='calls').fillna(0)
            fig = px.line(daily_usage_pivot, labels={"value": "API Calls", "created_at": "Date"})
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Breakdown by API Provider & Type")
            provider_type_stats = df.groupby(["api_provider", "api_type"]).agg({
                "id": "count", 
                "tokens_used": "sum", 
                "cost_usd": "sum"
            }).rename(columns={"id": "calls"}).reset_index()
            fig = px.bar(provider_type_stats, x="api_provider", y="calls", color="api_type", 
                         labels={"calls": "Number of API Calls", "api_provider": "API Provider"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(provider_type_stats)
            st.subheader("Per-User Usage & Cost")
            st.dataframe(df.groupby("user_id").agg({"id": "count", "tokens_used": "sum", "cost_usd": "sum"}).rename(columns={"id": "calls"}))
            st.subheader("Recent API Calls")
            st.dataframe(df.sort_values("created_at", ascending=False).head(50))
        else:
            st.info("No API usage logs found for the selected filters.")
    with tab2:
        st.subheader("Project Overview")
        try:
            projects_data, _ = client.table("projects").select("*").execute()
            projects_df = pd.DataFrame(projects_data[1] if projects_data and len(projects_data) > 1 else [])
            if not projects_df.empty:
                project_stats = []
                for _, project in projects_df.iterrows():
                    docs_data, _ = client.table("documents").select("id").eq("project_id", project["id"]).execute()
                    doc_count = len(docs_data[1]) if docs_data and len(docs_data) > 1 else 0
                    project_stats.append({
                        "project_id": project["id"],
                        "project_name": project["name"],
                        "description": project["description"] or "No description",
                        "created_at": project["created_at"],
                        "document_count": doc_count
                    })
                project_stats_df = pd.DataFrame(project_stats)
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(project_stats_df, x="project_name", y="document_count", 
                                title="Documents per Project",
                                labels={"project_name": "Project", "document_count": "Number of Documents"})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    project_stats_df["created_at"] = pd.to_datetime(project_stats_df["created_at"])
                    project_stats_df = project_stats_df.sort_values("created_at")
                    fig = px.timeline(project_stats_df, x_start="created_at", y="project_name", 
                                    title="Project Creation Timeline")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(project_stats_df[["project_name", "description", "document_count", "created_at"]])
            else:
                st.info("No projects found.")
        except Exception as e:
            st.error(f"Error fetching project data: {e}")
    with tab3:
        st.subheader("Document Statistics")
        doc_query = client.table("documents").select("*")
        if project_id_filter:
            doc_query = doc_query.eq("project_id", project_id_filter)
        try:
            docs_data, _ = doc_query.execute()
            docs_df = pd.DataFrame(docs_data[1] if docs_data and len(docs_data) > 1 else [])
            if not docs_df.empty:
                docs_df["created_at"] = pd.to_datetime(docs_df["created_at"])
                docs_df["month"] = docs_df["created_at"].dt.strftime("%Y-%m")
                docs_by_month = docs_df.groupby("month").size().reset_index(name="count")
                fig = px.bar(docs_by_month, x="month", y="count", 
                            title="Documents Processed per Month",
                            labels={"month": "Month", "count": "Number of Documents"})
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Total Documents", len(docs_df))
                avg_pages = docs_df["num_pages"].mean() if "num_pages" in docs_df.columns else 0
                st.metric("Average Pages per Document", f"{avg_pages:.1f}")
                version_counts = docs_df["version"].value_counts().reset_index()
                version_counts.columns = ["Version", "Count"]
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(version_counts, values="Count", names="Version", 
                                title="Document Versions")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("Recently Added Documents")
                    recent_docs = docs_df.sort_values("created_at", ascending=False).head(5)
                    for _, doc in recent_docs.iterrows():
                        st.write(f"**{doc['filename']}** - {doc['created_at'].strftime('%Y-%m-%d %H:%M')}")
                st.dataframe(docs_df[["filename", "num_pages", "version", "created_at"]])
            else:
                st.info("No documents found matching the selected filters.")
        except Exception as e:
            st.error(f"Error fetching document data: {e}")
    with tab4:
        st.subheader("Cost Analysis")
        if not df.empty and 'cost_usd' in df.columns:
            cost_by_type = df.groupby("api_type")["cost_usd"].sum().reset_index()
            fig = px.pie(cost_by_type, values="cost_usd", names="api_type", 
                        title="Cost Distribution by API Type")
            st.plotly_chart(fig, use_container_width=True)
            df['date'] = df['created_at'].dt.date
            cost_by_date = df.groupby(['date', 'api_type'])['cost_usd'].sum().reset_index()
            fig = px.line(cost_by_date, x="date", y="cost_usd", color="api_type",
                        title="Daily API Costs",
                        labels={"date": "Date", "cost_usd": "Cost (USD)", "api_type": "API Type"})
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Cost Efficiency")
            col1, col2 = st.columns(2)
            with col1:
                token_cost = df.groupby("api_type").agg({
                    "tokens_used": "sum",
                    "cost_usd": "sum"
                }).reset_index()
                token_cost["cost_per_1k_tokens"] = token_cost["cost_usd"] / token_cost["tokens_used"] * 1000
                fig = px.bar(token_cost, x="api_type", y="cost_per_1k_tokens",
                            title="Cost per 1,000 Tokens by API Type",
                            labels={"api_type": "API Type", "cost_per_1k_tokens": "USD per 1K Tokens"})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                daily_cost = df.groupby(df['created_at'].dt.date)["cost_usd"].sum().reset_index()
                daily_cost.columns = ["date", "cost"]
                if len(daily_cost) > 1:
                    daily_cost['day_number'] = range(len(daily_cost))
                    import numpy as np
                    from sklearn.linear_model import LinearRegression
                    try:
                        X = daily_cost['day_number'].values.reshape(-1, 1)
                        y = daily_cost['cost'].values
                        model = LinearRegression().fit(X, y)
                        daily_cost['trend'] = model.predict(X)
                        future_days = pd.DataFrame({
                            'day_number': range(len(daily_cost), len(daily_cost) + 30),
                            'date': [daily_cost['date'].iloc[-1] + timedelta(days=i+1) for i in range(30)]
                        })
                        future_days['trend'] = model.predict(future_days['day_number'].values.reshape(-1, 1))
                        projection_df = pd.concat([
                            daily_cost[['date', 'cost', 'trend']],
                            future_days[['date', 'trend']].rename(columns={'trend': 'projected_cost'})
                        ])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=daily_cost['date'], y=daily_cost['cost'], 
                                                mode='markers', name='Actual Cost'))
                        fig.add_trace(go.Scatter(x=projection_df['date'], y=projection_df['trend'], 
                                                mode='lines', name='Trend'))
                        fig.add_trace(go.Scatter(x=future_days['date'], y=future_days['trend'], 
                                                mode='lines', name='Projection', line=dict(dash='dash')))
                        fig.update_layout(title="Cost Projection (Next 30 Days)",
                                        xaxis_title="Date",
                                        yaxis_title="Cost (USD)")
                        st.plotly_chart(fig, use_container_width=True)
                        avg_daily_cost = daily_cost['cost'].mean()
                        projected_monthly = avg_daily_cost * 30
                        st.metric("Projected Monthly Cost", f"${projected_monthly:.2f}")
                    except Exception as e:
                        st.error(f"Error calculating projection: {e}")
                else:
                    st.info("Not enough data for cost projection.")
        else:
            st.info("No cost data available for the selected filters.")

    # Entities tab
    with tab5:
        st.subheader("🌾 Agricultural Entities Analysis")
        
        try:
            # Query agricultural entities
            entities_query = client.table("agricultural_entities").select("*")
            if project_id_filter:
                # Filter by project through documents
                project_docs_data, _ = client.table("documents").select("id").eq("project_id", project_id_filter).execute()
                project_doc_ids = [doc["id"] for doc in project_docs_data[1]] if project_docs_data and len(project_docs_data) > 1 else []
                if project_doc_ids:
                    entities_query = entities_query.in_("document_id", project_doc_ids)
                else:
                    st.warning("No documents found in the selected project.")
                    
            entities_data, _ = entities_query.execute()
            entities_df = pd.DataFrame(entities_data[1] if entities_data and len(entities_data) > 1 else [])
            
            if not entities_df.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entities", len(entities_df))
                with col2:
                    unique_types = entities_df["entity_type"].nunique()
                    st.metric("Entity Types", unique_types)
                with col3:
                    unique_docs = entities_df["document_id"].nunique()
                    st.metric("Documents with Entities", unique_docs)
                with col4:
                    avg_confidence = entities_df["confidence_score"].mean() if "confidence_score" in entities_df.columns else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                # Entity type distribution
                st.subheader("📊 Entity Type Distribution")
                type_counts = entities_df["entity_type"].value_counts().reset_index()
                type_counts.columns = ["Entity Type", "Count"]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(type_counts, values="Count", names="Entity Type", 
                                title="Distribution of Entity Types")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(type_counts, x="Entity Type", y="Count", 
                                title="Entity Count by Type",
                                labels={"Entity Type": "Type", "Count": "Number of Entities"})
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence score analysis
                if "confidence_score" in entities_df.columns:
                    st.subheader("🎯 Confidence Score Analysis")
                    fig = px.histogram(entities_df, x="confidence_score", nbins=20,
                                     title="Distribution of Entity Confidence Scores",
                                     labels={"confidence_score": "Confidence Score", "count": "Number of Entities"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence by entity type
                    conf_by_type = entities_df.groupby("entity_type")["confidence_score"].mean().reset_index()
                    fig = px.bar(conf_by_type, x="entity_type", y="confidence_score",
                                title="Average Confidence Score by Entity Type",
                                labels={"entity_type": "Entity Type", "confidence_score": "Avg Confidence"})
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recent entities
                st.subheader("🕒 Recently Extracted Entities")
                if "created_at" in entities_df.columns:
                    entities_df["created_at"] = pd.to_datetime(entities_df["created_at"])
                    recent_entities = entities_df.sort_values("created_at", ascending=False).head(10)
                    display_cols = ["entity_value", "entity_type", "created_at"]
                    if "confidence_score" in entities_df.columns:
                        display_cols.append("confidence_score")
                    st.dataframe(recent_entities[display_cols])
                else:
                    # Show sample entities without date sorting
                    display_cols = ["entity_value", "entity_type"]
                    if "confidence_score" in entities_df.columns:
                        display_cols.append("confidence_score")
                    st.dataframe(entities_df[display_cols].head(10))
                
                # Top entities by type
                st.subheader("🏆 Top Entities by Type")
                entity_types = entities_df["entity_type"].unique()
                
                # Create tabs for each entity type
                if len(entity_types) > 1:
                    entity_type_tabs = st.tabs([f"{etype} ({len(entities_df[entities_df['entity_type']==etype])})" 
                                               for etype in entity_types[:5]])  # Limit to 5 types for UI
                    
                    for i, etype in enumerate(entity_types[:5]):
                        with entity_type_tabs[i]:
                            type_entities = entities_df[entities_df["entity_type"] == etype]
                            
                            # Count occurrences of each entity value
                            entity_counts = type_entities["entity_value"].value_counts().head(20).reset_index()
                            entity_counts.columns = ["Entity", "Occurrences"]
                            
                            if len(entity_counts) > 0:
                                fig = px.bar(entity_counts, x="Entity", y="Occurrences",
                                           title=f"Top {etype} Entities",
                                           labels={"Entity": "Entity Value", "Occurrences": "Count"})
                                fig.update_xaxis(tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.dataframe(entity_counts)
                            else:
                                st.info(f"No {etype} entities found.")
                
                # Full entities table (paginated)
                st.subheader("📋 All Entities")
                st.info("Showing entities data - you can search and filter using the controls below.")
                
                # Add search functionality
                search_term = st.text_input("🔍 Search entities by value")
                type_filter = st.selectbox("Filter by type", ["All"] + list(entities_df["entity_type"].unique()))
                
                # Apply filters
                filtered_entities = entities_df.copy()
                if search_term:
                    filtered_entities = filtered_entities[
                        filtered_entities["entity_value"].str.contains(search_term, case=False, na=False)
                    ]
                if type_filter != "All":
                    filtered_entities = filtered_entities[filtered_entities["entity_type"] == type_filter]
                
                st.write(f"Showing {len(filtered_entities)} of {len(entities_df)} entities")
                
                # Display with pagination
                items_per_page = 50
                total_pages = len(filtered_entities) // items_per_page + (1 if len(filtered_entities) % items_per_page > 0 else 0)
                
                if total_pages > 1:
                    page = st.selectbox("Page", range(1, total_pages + 1))
                    start_idx = (page - 1) * items_per_page
                    end_idx = start_idx + items_per_page
                    display_entities = filtered_entities.iloc[start_idx:end_idx]
                else:
                    display_entities = filtered_entities
                
                # Select columns to display
                available_cols = list(entities_df.columns)
                default_cols = ["entity_value", "entity_type", "document_id"]
                if "confidence_score" in available_cols:
                    default_cols.append("confidence_score")
                if "created_at" in available_cols:
                    default_cols.append("created_at")
                
                selected_cols = st.multiselect(
                    "Select columns to display", 
                    available_cols, 
                    default=[col for col in default_cols if col in available_cols]
                )
                
                if selected_cols:
                    st.dataframe(display_entities[selected_cols], use_container_width=True)
                else:
                    st.warning("Please select at least one column to display.")
                    
            else:
                st.info("No agricultural entities found for the selected filters.")
                
        except Exception as e:
            st.error(f"Error fetching entities data: {e}")
            st.write("Debug info:", str(e))

    # Relationships tab
    with tab6:
        st.subheader("🔗 Agricultural Relationships Analysis")
        
        try:
            # Query agricultural relationships
            relationships_query = client.table("agricultural_relationships").select("*")
            if project_id_filter:
                # Filter by project through documents
                project_docs_data, _ = client.table("documents").select("id").eq("project_id", project_id_filter).execute()
                project_doc_ids = [doc["id"] for doc in project_docs_data[1]] if project_docs_data and len(project_docs_data) > 1 else []
                if project_doc_ids:
                    relationships_query = relationships_query.in_("document_id", project_doc_ids)
                else:
                    st.warning("No documents found in the selected project.")
                    
            relationships_data, _ = relationships_query.execute()
            relationships_df = pd.DataFrame(relationships_data[1] if relationships_data and len(relationships_data) > 1 else [])
            
            if not relationships_df.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Relationships", len(relationships_df))
                with col2:
                    unique_types = relationships_df["relationship_type"].nunique()
                    st.metric("Relationship Types", unique_types)
                with col3:
                    unique_docs = relationships_df["document_id"].nunique()
                    st.metric("Documents with Relationships", unique_docs)
                with col4:
                    avg_confidence = relationships_df["confidence_score"].mean() if "confidence_score" in relationships_df.columns else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                # Relationship type distribution
                st.subheader("📊 Relationship Type Distribution")
                type_counts = relationships_df["relationship_type"].value_counts().reset_index()
                type_counts.columns = ["Relationship Type", "Count"]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(type_counts, values="Count", names="Relationship Type", 
                                title="Distribution of Relationship Types")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(type_counts, x="Relationship Type", y="Count", 
                                title="Relationship Count by Type",
                                labels={"Relationship Type": "Type", "Count": "Number of Relationships"})
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Network analysis
                st.subheader("🕸️ Network Analysis")
                
                # Get entity data for network visualization
                try:
                    entities_query = client.table("agricultural_entities").select("id, entity_value, entity_type")
                    if project_id_filter and project_doc_ids:
                        entities_query = entities_query.in_("document_id", project_doc_ids)
                    entities_data, _ = entities_query.execute()
                    entities_df = pd.DataFrame(entities_data[1] if entities_data and len(entities_data) > 1 else [])
                    
                    if not entities_df.empty:
                        # Create entity lookup
                        entity_lookup = entities_df.set_index('id')['entity_value'].to_dict()
                        entity_type_lookup = entities_df.set_index('id')['entity_type'].to_dict()
                        
                        # Add entity names to relationships
                        relationships_display = relationships_df.copy()
                        relationships_display['source_entity'] = relationships_display['source_entity_id'].map(entity_lookup)
                        relationships_display['target_entity'] = relationships_display['target_entity_id'].map(entity_lookup)
                        relationships_display['source_type'] = relationships_display['source_entity_id'].map(entity_type_lookup)
                        relationships_display['target_type'] = relationships_display['target_entity_id'].map(entity_type_lookup)
                        
                        # Remove rows where entity lookup failed
                        relationships_display = relationships_display.dropna(subset=['source_entity', 'target_entity'])
                        
                        if not relationships_display.empty:
                            # Network statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                unique_nodes = set(relationships_display['source_entity'].tolist() + relationships_display['target_entity'].tolist())
                                st.metric("Unique Entities in Network", len(unique_nodes))
                            with col2:
                                avg_degree = len(relationships_display) * 2 / len(unique_nodes) if len(unique_nodes) > 0 else 0
                                st.metric("Average Node Degree", f"{avg_degree:.1f}")
                            with col3:
                                density = len(relationships_display) / (len(unique_nodes) * (len(unique_nodes) - 1) / 2) if len(unique_nodes) > 1 else 0
                                st.metric("Network Density", f"{density:.3f}")
                            
                            # Most connected entities
                            st.subheader("🌟 Most Connected Entities")
                            source_counts = relationships_display['source_entity'].value_counts()
                            target_counts = relationships_display['target_entity'].value_counts()
                            total_counts = (source_counts + target_counts).fillna(0).sort_values(ascending=False).head(10)
                            
                            if len(total_counts) > 0:
                                connected_df = pd.DataFrame({
                                    'Entity': total_counts.index,
                                    'Connections': total_counts.values
                                })
                                
                                fig = px.bar(connected_df, x="Entity", y="Connections",
                                           title="Top 10 Most Connected Entities",
                                           labels={"Entity": "Entity Name", "Connections": "Number of Connections"})
                                fig.update_xaxis(tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Sample relationships table
                        st.subheader("🔗 Sample Relationships")
                        display_cols = ['source_entity', 'relationship_type', 'target_entity']
                        if 'confidence_score' in relationships_display.columns:
                            display_cols.append('confidence_score')
                        if 'source_type' in relationships_display.columns:
                            display_cols.extend(['source_type', 'target_type'])
                        
                        sample_relationships = relationships_display[display_cols].head(20)
                        st.dataframe(sample_relationships, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"Could not load entity data for network analysis: {e}")
                    # Fallback to showing relationships without entity names
                    st.subheader("🔗 Relationships (Entity IDs)")
                    sample_cols = ['source_entity_id', 'relationship_type', 'target_entity_id']
                    if 'confidence_score' in relationships_df.columns:
                        sample_cols.append('confidence_score')
                    st.dataframe(relationships_df[sample_cols].head(20))
                
                # Confidence score analysis
                if "confidence_score" in relationships_df.columns:
                    st.subheader("🎯 Relationship Confidence Analysis")
                    fig = px.histogram(relationships_df, x="confidence_score", nbins=20,
                                     title="Distribution of Relationship Confidence Scores",
                                     labels={"confidence_score": "Confidence Score", "count": "Number of Relationships"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence by relationship type
                    conf_by_type = relationships_df.groupby("relationship_type")["confidence_score"].mean().reset_index()
                    fig = px.bar(conf_by_type, x="relationship_type", y="confidence_score",
                                title="Average Confidence Score by Relationship Type",
                                labels={"relationship_type": "Relationship Type", "confidence_score": "Avg Confidence"})
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Relationship patterns
                st.subheader("🔍 Relationship Patterns")
                pattern_analysis = relationships_df.groupby("relationship_type").size().reset_index(name="count")
                pattern_analysis = pattern_analysis.sort_values("count", ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Relationship Types:**")
                    for _, row in pattern_analysis.head(10).iterrows():
                        st.write(f"• {row['relationship_type']}: {row['count']} relationships")
                
                with col2:
                    # Relationship type combinations (if multiple documents)
                    if relationships_df["document_id"].nunique() > 1:
                        doc_patterns = relationships_df.groupby(["document_id", "relationship_type"]).size().reset_index(name="count")
                        st.write("**Relationship patterns vary across documents**")
                        st.write(f"Documents: {relationships_df['document_id'].nunique()}")
                        st.write(f"Avg relationships per document: {len(relationships_df) / relationships_df['document_id'].nunique():.1f}")
                    else:
                        st.write("**Single Document Analysis**")
                        st.write(f"All relationships from one document")
                        st.write(f"Total relationships: {len(relationships_df)}")
                
                # Full relationships table (paginated)
                st.subheader("📋 All Relationships")
                st.info("Showing relationships data - you can search and filter using the controls below.")
                
                # Add search and filter functionality
                search_term = st.text_input("🔍 Search relationships (source, target, or type)")
                type_filter = st.selectbox("Filter by relationship type", ["All"] + list(relationships_df["relationship_type"].unique()))
                
                # Apply filters
                filtered_relationships = relationships_df.copy()
                if search_term:
                    # Search in multiple columns if available
                    search_mask = False
                    if 'source_entity' in relationships_display.columns:
                        search_mask |= relationships_display["source_entity"].str.contains(search_term, case=False, na=False)
                    if 'target_entity' in relationships_display.columns:
                        search_mask |= relationships_display["target_entity"].str.contains(search_term, case=False, na=False)
                    search_mask |= relationships_df["relationship_type"].str.contains(search_term, case=False, na=False)
                    filtered_relationships = relationships_df[search_mask]
                
                if type_filter != "All":
                    filtered_relationships = filtered_relationships[filtered_relationships["relationship_type"] == type_filter]
                
                st.write(f"Showing {len(filtered_relationships)} of {len(relationships_df)} relationships")
                
                # Display with pagination
                items_per_page = 50
                total_pages = len(filtered_relationships) // items_per_page + (1 if len(filtered_relationships) % items_per_page > 0 else 0)
                
                if total_pages > 1:
                    page = st.selectbox("Page", range(1, total_pages + 1), key="rel_page")
                    start_idx = (page - 1) * items_per_page
                    end_idx = start_idx + items_per_page
                    display_relationships = filtered_relationships.iloc[start_idx:end_idx]
                else:
                    display_relationships = filtered_relationships
                
                # Select columns to display
                available_cols = list(relationships_df.columns)
                default_cols = ["source_entity_id", "relationship_type", "target_entity_id"]
                if "confidence_score" in available_cols:
                    default_cols.append("confidence_score")
                if "created_at" in available_cols:
                    default_cols.append("created_at")
                
                selected_cols = st.multiselect(
                    "Select columns to display", 
                    available_cols, 
                    default=[col for col in default_cols if col in available_cols],
                    key="rel_cols"
                )
                
                if selected_cols:
                    st.dataframe(display_relationships[selected_cols], use_container_width=True)
                else:
                    st.warning("Please select at least one column to display.")
                    
            else:
                st.info("No agricultural relationships found for the selected filters.")
                
        except Exception as e:
            st.error(f"Error fetching relationships data: {e}")
            st.write("Debug info:", str(e))

if __name__ == "__main__" or (hasattr(sys, "argv") and any("streamlit" in arg for arg in sys.argv)):
    run_dashboard() 