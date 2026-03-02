import pandas as pd
import numpy as np
import os
import argparse
import json
import random

def generate_student_data(num_records=2000, output_path="data/placement_data.csv"):
    """
    Generates synthetic student data for placement prediction.
    """
    print(f"Generating {num_records} synthetic student records...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Academic Features
    cgpa = np.random.uniform(5.0, 10.0, num_records)
    backlogs = np.random.randint(0, 6, num_records)
    branches = ['CSE', 'IT', 'ECE', 'MECH', 'CIVIL', 'CYBER', 'AIDS', 'AIML']
    branch = np.random.choice(branches, num_records, p=[0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10])
    tenth_percentage = np.clip(np.random.normal(80, 10, num_records), 60, 100)
    twelfth_percentage = np.clip(tenth_percentage + np.random.normal(0, 5, num_records), a_min=60, a_max=100)

    # Skill Features Pools
    core_tech_skills = ['python', 'java', 'c++', 'dsa', 'machine learning', 'sql', 'cloud', 'react', 'node', 'devops', 'cybersecurity', 'ai', 'data science']
    supporting_tech_skills = ['html', 'css', 'git', 'docker', 'excel', 'testing', 'figma']
    soft_skills = ['communication', 'leadership', 'teamwork', 'presentation', 'management']
    proficiency_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
    
    # Generate dynamic skills list for each student
    dynamic_skills_list = []
    dynamic_skill_scores = []
    
    for _ in range(num_records):
        num_skills = np.random.randint(3, 8)
        student_skills = []
        score_sum = 0
        
        # Pick skills from pools avoiding duplicates
        all_picks = random.sample(core_tech_skills, min(len(core_tech_skills), np.random.randint(1, 4))) + \
                    random.sample(supporting_tech_skills, min(len(supporting_tech_skills), np.random.randint(1, 3))) + \
                    random.sample(soft_skills, min(len(soft_skills), np.random.randint(1, 3)))
        
        selected_skills = random.sample(all_picks, min(len(all_picks), num_skills))
        
        for skill in selected_skills:
            level = np.random.choice(proficiency_levels, p=[0.2, 0.4, 0.3, 0.1])
            student_skills.append({"skill": skill, "level": level})
            
            # Temporary numeric score computation for generator probability calculations
            level_num = proficiency_levels.index(level) + 1
            if skill in core_tech_skills:
                score_sum += level_num * 1.5
            elif skill in supporting_tech_skills:
                score_sum += level_num * 1.2
            elif skill in soft_skills:
                score_sum += level_num * 0.8
            else:
                score_sum += level_num * 1.0 # Unknown
                
        dynamic_skills_list.append(json.dumps(student_skills))
        # Expected max score is roughly 7 skills * 4 * 1.5 = 42
        dynamic_skill_scores.append(score_sum)
        
    dynamic_skill_scores = np.array(dynamic_skill_scores)

    dsa_score = np.clip(np.random.normal(65, 20, num_records), 0, 100)
    projects_count = np.random.randint(0, 7, num_records)
    certifications = np.random.randint(0, 6, num_records)
    internships = np.random.randint(0, 4, num_records)
    
    # Add correlation: higher CGPA often means higher DSA and more projects
    dsa_score += (cgpa - 7.5) * 5
    dsa_score = np.clip(dsa_score, 0, 100)
    
    projects_count += np.where(cgpa > 8.0, np.random.randint(0, 2, num_records), 0)
    projects_count = np.clip(projects_count, 0, 6)

    # Activity Features
    hackathons = np.random.randint(0, 6, num_records)
    coding_rating = np.clip(np.random.normal(1200, 300, num_records), 800, 2000)
    clubs = np.random.randint(0, 4, num_records)
    leadership_roles = np.random.randint(0, 3, num_records)
    
    # Add correlation: higher DSA often means higher coding rating and more hackathons
    coding_rating += (dsa_score - 60) * 5
    coding_rating = np.clip(coding_rating, 800, 2000)

    # Calculate Probability based on weighted conditions
    # Base probability
    prob = np.full(num_records, 0.3)
    
    # Positive factors
    prob += (cgpa - 5.0) / 5.0 * 0.4  # Max 0.4 from CGPA
    prob += (dsa_score / 100.0) * 0.2 # Max 0.2 from DSA
    prob += (internships / 3.0) * 0.15 # Max 0.15 from Internships
    prob += (projects_count / 6.0) * 0.1 # Max 0.1 from Projects
    prob += (coding_rating - 800) / 1200 * 0.1 # Max 0.1 from Coding Rating
    prob += (dynamic_skill_scores / 42.0) * 0.15 # Max 0.15 from Dynamic Skills

    # Branch modifiers
    branch_modifiers = {'CSE': 0.1, 'IT': 0.08, 'ECE': 0.05, 'MECH': 0.0, 'CIVIL': -0.05, 'CYBER': 0.1, 'AIDS': 0.1, 'AIML': 0.1}
    prob += np.array([branch_modifiers[b] for b in branch])
    
    # Negative factors
    prob -= (backlogs / 5.0) * 0.4 # Max 0.4 penalty for backlogs
    
    # Add noise
    noise = np.random.normal(0, 0.1, num_records)
    prob += noise
    
    # Final probability clip
    prob = np.clip(prob, 0, 1)
    
    # Target Variable: Placed (0 or 1)
    # Use probability to determine placement (stochastic)
    placed = np.random.binomial(1, prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CGPA': np.round(cgpa, 2),
        'Backlogs': backlogs,
        'Branch': branch,
        '10th_Percentage': np.round(tenth_percentage, 2),
        '12th_Percentage': np.round(twelfth_percentage, 2),
        'skills': dynamic_skills_list,
        'DSA_Score': np.round(dsa_score, 2),
        'Projects_Count': projects_count,
        'Certifications': certifications,
        'Internships': internships,
        'Hackathons_Participated': hackathons,
        'Coding_Contest_Rating': np.round(coding_rating, 0).astype(int),
        'Clubs': clubs,
        'Leadership_Roles': leadership_roles,
        'Placed': placed
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print("\nDataset Strategy Statistics:")
    print(df['Placed'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic student placement data.')
    parser.add_argument('--num_records', type=int, default=2000, help='Number of records to generate')
    parser.add_argument('--output', type=str, default='data/placement_data.csv', help='Output file path')
    args = parser.parse_args()
    
    generate_student_data(args.num_records, args.output)
