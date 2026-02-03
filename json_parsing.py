import json

def load_scene_data():
    """scene_groups_with_goal_label.json 파일을 로드합니다."""
    with open("scene_groups_with_goal_label.json", "r") as f:
        data = json.load(f)
    return data

def get_scene_data_by_group_id(group_id):
    """특정 group_id에 해당하는 scene 데이터를 반환합니다."""
    data = load_scene_data()
    for v in data:
        if v.get("group_id") == group_id:
            return {
                "floorplan": v.get("floorplan", []),
                "objects": v.get("objects", []),
                "people": v.get("people", []),
                "task": v.get("task", "")
            }
    return None

def get_goals_data_by_group_id(group_id):
    """특정 group_id에 해당하는 goal_labels 데이터를 반환합니다."""
    data = load_scene_data()
    for v in data:
        if v.get("group_id") == group_id:
            goal_labels = v.get("goal_label", [])
            # goal_label이 문자열인 경우를 처리
            if isinstance(goal_labels, str):
                goal_labels = [goal_labels]
            return goal_labels
    return []

def get_all_group_ids():
    """모든 group_id 목록을 반환합니다."""
    data = load_scene_data()
    return [v.get("group_id") for v in data if v.get("group_id") is not None]

def get_group_info(group_id):
    """특정 group_id의 모든 정보를 반환합니다."""
    data = load_scene_data()
    for v in data:
        if v.get("group_id") == group_id:
            return {
                "group_id": v.get("group_id"),
                "count": v.get("count"),
                "floorplan": v.get("floorplan", []),
                "objects": v.get("objects", []),
                "people": v.get("people", []),
                "task": v.get("task", ""),
                "goal_label": v.get("goal_label", [])
            }
    return None

# 테스트용 코드
if __name__ == "__main__":
    # 모든 group_id 출력
    group_ids = get_all_group_ids()
    print(f"Available group IDs: {group_ids}")
    
    # 첫 번째 그룹 정보 출력
    if group_ids:
        first_group = get_group_info(group_ids[0])
        print(f"\nFirst group info: {first_group}")
        
        scene_data = get_scene_data_by_group_id(group_ids[0])
        goals_data = get_goals_data_by_group_id(group_ids[0])
        print(f"\nScene data: {scene_data}")
        print(f"Goals data: {goals_data}")
    
    
    