"""Manual smoke tests for search_tasks (no LLM or LangGraph needed)."""

from pprint import pprint
from search_tools import * 
cases = {
    'Tasks assigned to Ali Saberi': {'assignee_name': 'رضا محمدی'},
    'Open tasks of support department': {'department': 'پشتیبانی', 'status': 'Open'},
    'High-priority tasks': {'priority': 'High'},
    'Next page of high-priority tasks': {'priority': 'High', 'limit': 10, 'offset': 10},
}

for label, filters in cases.items():
    print(f'\n--- {label} ---')
    result = search_tasks.invoke(filters)
    pprint({key: value for key, value in result.items() if key != 'tasks'}, sort_dicts=False)
    pprint(result['tasks'][:3], sort_dicts=False)



from search_tools import search_tasks

names = [
    "علی صابری",
    "رضا محمدی",
    "مریم کریمی"
]

for name in names:
    result = search_tasks.invoke({
        "assignee_name": name
    })

    print("\nRequested:", name)
    print("Applied:", result["applied_filters"])
    print("Total:", result["total_count"])

    for task in result["tasks"]:
        print(
            task["id"],
            task["assignee_name"],
            task["title"]
        )