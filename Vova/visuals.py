import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

df = pd.read_csv('pfas_tenders_output/pfas_tenders_accurate.csv')
df['publication_date'] = pd.to_datetime(df['publication_date'])
df['year'] = df['publication_date'].dt.year

print(f"Loaded {len(df)} tenders")

os.makedirs('visualizations', exist_ok=True)


# Temporal trends
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

yearly = df['year'].value_counts().sort_index()
ax1.bar(yearly.index, yearly.values, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('PFAS Tenders by Year', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Tenders')
ax1.grid(axis='y', alpha=0.3)

for year, count in yearly.items():
    ax1.text(year, count + 1, str(count), ha='center', va='bottom', fontweight='bold')

cumulative = yearly.cumsum()
ax2.plot(cumulative.index, cumulative.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
ax2.fill_between(cumulative.index, cumulative.values, alpha=0.3, color='darkgreen')
ax2.set_title('Cumulative PFAS Tenders Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Cumulative Number of Tenders')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/1_temporal_trends.png', dpi=300, bbox_inches='tight')
plt.close()


# Contract types
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

contract_counts = df['contract_type'].value_counts()
colors = ['#66c2a5', '#fc8d62', '#8da0cb']
ax1.pie(contract_counts.values, labels=contract_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Contract Type Distribution', fontsize=14, fontweight='bold')

contract_by_year = pd.crosstab(df['year'], df['contract_type'])
contract_by_year.plot(kind='bar', stacked=True, ax=ax2, color=colors, edgecolor='black', alpha=0.8)
ax2.set_title('Contract Types Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Tenders')
ax2.legend(title='Contract Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/2_contract_types.png', dpi=300, bbox_inches='tight')
plt.close()


# Top organizations
fig, ax = plt.subplots(figsize=(12, 8))

top_orgs = df['organization'].value_counts().head(15)
ax.barh(range(len(top_orgs)), top_orgs.values, color='coral', edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(top_orgs)))
ax.set_yticklabels(top_orgs.index, fontsize=10)
ax.set_xlabel('Number of Tenders', fontsize=12)
ax.set_title('Top 15 Organizations Procuring PFAS Work', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(top_orgs.values):
    ax.text(v + 0.2, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/3_top_organizations.png', dpi=300, bbox_inches='tight')
plt.close()


# Organization types
def categorize_org(org):
    if pd.isna(org):
        return 'Unknown'
    org_lower = org.lower()
    if 'gemeente' in org_lower:
        return 'Municipality'
    elif 'provincie' in org_lower:
        return 'Province'
    elif 'waterschap' in org_lower or 'hoogheemraadschap' in org_lower:
        return 'Water Board'
    elif 'rijk' in org_lower or 'ministerie' in org_lower:
        return 'National Government'
    elif 'brandweer' in org_lower:
        return 'Fire Department'
    elif 'university' in org_lower or 'universiteit' in org_lower:
        return 'Research Institution'
    else:
        return 'Other'

df['org_type'] = df['organization'].apply(categorize_org)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

org_types = df['org_type'].value_counts()
colors_org = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
ax1.pie(org_types.values, labels=org_types.index, autopct='%1.1f%%',
        startangle=90, colors=colors_org[:len(org_types)], 
        textprops={'fontsize': 10, 'fontweight': 'bold'})
ax1.set_title('Tenders by Organization Type', fontsize=14, fontweight='bold')

procedure_types = df['procedure_type'].value_counts().head(5)
ax2.bar(range(len(procedure_types)), procedure_types.values, color='mediumpurple', edgecolor='black', alpha=0.8)
ax2.set_xticks(range(len(procedure_types)))
ax2.set_xticklabels(procedure_types.index, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Number of Tenders')
ax2.set_title('Top 5 Procurement Procedures', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for i, v in enumerate(procedure_types.values):
    ax2.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/4_organization_types.png', dpi=300, bbox_inches='tight')
plt.close()


# PFAS mentions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(df['pfas_mentions'], bins=range(0, df['pfas_mentions'].max()+2), 
         color='salmon', edgecolor='black', alpha=0.7)
ax1.set_title('Distribution of PFAS Mentions per Tender', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of PFAS Mentions')
ax1.set_ylabel('Frequency')
ax1.axvline(df['pfas_mentions'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["pfas_mentions"].mean():.1f}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

pfas_by_year = df.groupby('year')['pfas_mentions'].mean()
ax2.plot(pfas_by_year.index, pfas_by_year.values, marker='o', linewidth=2, markersize=8, color='darkred')
ax2.fill_between(pfas_by_year.index, pfas_by_year.values, alpha=0.3, color='darkred')
ax2.set_title('Average PFAS Mentions per Tender Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Average PFAS Mentions')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/5_pfas_mentions.png', dpi=300, bbox_inches='tight')
plt.close()


# European vs National
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

european_counts = df['european'].value_counts()
labels = ['European' if x else 'National' for x in european_counts.index]
ax1.pie(european_counts.values, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=['#4575b4', '#d73027'], textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('European vs. National Procurement', fontsize=14, fontweight='bold')

european_by_year = df.groupby('year')['european'].sum()
total_by_year = df.groupby('year').size()
pct_european = (european_by_year / total_by_year * 100)

ax2.bar(pct_european.index, pct_european.values, color='#4575b4', edgecolor='black', alpha=0.7)
ax2.axhline(y=81.8, color='red', linestyle='--', linewidth=2, label='Overall Average (81.8%)')
ax2.set_title('Percentage of European Tenders by Year', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('% European Tenders')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('visualizations/6_european_national.png', dpi=300, bbox_inches='tight')
plt.close()


# Summary dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('PFAS Tender Analysis Dashboard (2017-2025)', fontsize=18, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
metrics_text = f"""
KEY METRICS

Total Tenders: {len(df)}
Date Range: 2017-2025
European: {df['european'].sum()} ({df['european'].sum()/len(df)*100:.1f}%)

Contract Types:
  Services: {(df['contract_type']=='Diensten').sum()} (56%)
  Works: {(df['contract_type']=='Werken').sum()} (39%)
  Supplies: {(df['contract_type']=='Leveringen').sum()} (5%)

Avg PFAS Mentions: {df['pfas_mentions'].mean():.1f}
"""
ax1.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2 = fig.add_subplot(gs[0, 1:])
yearly = df['year'].value_counts().sort_index()
ax2.bar(yearly.index, yearly.values, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_title('Tenders by Year', fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Count')
ax2.grid(axis='y', alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
contract_counts = df['contract_type'].value_counts()
ax3.pie(contract_counts.values, labels=contract_counts.index, autopct='%1.0f%%',
        startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
ax3.set_title('Contract Types', fontweight='bold')

ax4 = fig.add_subplot(gs[1, 1:])
top_orgs = df['organization'].value_counts().head(10)
ax4.barh(range(len(top_orgs)), top_orgs.values, color='coral', edgecolor='black', alpha=0.8)
ax4.set_yticks(range(len(top_orgs)))
ax4.set_yticklabels(top_orgs.index, fontsize=9)
ax4.set_xlabel('Number of Tenders')
ax4.set_title('Top 10 Organizations', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

ax5 = fig.add_subplot(gs[2, 0])
org_types = df['org_type'].value_counts()
ax5.pie(org_types.values, labels=org_types.index, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
ax5.set_title('Organization Types', fontweight='bold')

ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(df['pfas_mentions'], bins=range(0, min(df['pfas_mentions'].max()+2, 15)), 
         color='salmon', edgecolor='black', alpha=0.7)
ax6.set_title('PFAS Mentions Distribution', fontweight='bold')
ax6.set_xlabel('Mentions')
ax6.set_ylabel('Frequency')
ax6.grid(axis='y', alpha=0.3)

ax7 = fig.add_subplot(gs[2, 2])
european_counts = df['european'].value_counts()
ax7.pie(european_counts.values, labels=['European', 'National'], autopct='%1.1f%%',
        startangle=90, colors=['#4575b4', '#d73027'], textprops={'fontweight': 'bold'})
ax7.set_title('European vs National', fontweight='bold')

plt.savefig('visualizations/7_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("Done! All visualizations saved to visualizations/")
