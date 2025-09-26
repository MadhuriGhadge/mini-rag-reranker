import json
import pandas as pd
from typing import Dict, List
from search import SearchEngine
import time

class RAGEvaluator:
    def __init__(self):
        self.search_engine = SearchEngine()
    
    def load_test_questions(self, filepath: str = "test_questions.json") -> List[Dict]:
        """Load test questions from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def evaluate_question(self, question_data: Dict, mode: str, k: int = 5) -> Dict:
        """Evaluate a single question"""
        question = question_data['question']
        expected_keywords = [kw.lower() for kw in question_data['expected_keywords']]
        
        start_time = time.time()
        
        # Get search results
        result = self.search_engine.search_and_answer(question, k=k, mode=mode)
        
        response_time = time.time() - start_time
        
        # Calculate relevance score
        relevance_score = self.calculate_relevance_score(
            result['contexts'], 
            expected_keywords
        )
        
        # Calculate answer quality score
        answer_quality = self.calculate_answer_quality(
            result['answer'], 
            expected_keywords,
            result['abstained']
        )
        
        return {
            'question_id': question_data['id'],
            'question': question,
            'mode': mode,
            'abstained': result['abstained'],
            'abstain_reason': result['abstain_reason'],
            'top_score': result['contexts'][0]['score'] if result['contexts'] else 0.0,
            'relevance_score': relevance_score,
            'answer_quality': answer_quality,
            'response_time': response_time,
            'num_contexts': len(result['contexts'])
        }
    
    def calculate_relevance_score(self, contexts: List[Dict], expected_keywords: List[str]) -> float:
        """Calculate how relevant the retrieved contexts are"""
        if not contexts:
            return 0.0
        
        total_score = 0.0
        for i, context in enumerate(contexts):
            text_lower = context['text'].lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in expected_keywords if keyword in text_lower)
            keyword_ratio = keyword_matches / len(expected_keywords) if expected_keywords else 0
            
            # Weight by position (first result is more important)
            position_weight = 1.0 / (i + 1)
            
            # Weight by retrieval score
            retrieval_weight = context['score']
            
            context_score = keyword_ratio * position_weight * retrieval_weight
            total_score += context_score
        
        # Normalize by number of contexts
        return total_score / len(contexts)
    
    def calculate_answer_quality(self, answer: str, expected_keywords: List[str], abstained: bool) -> float:
        """Calculate answer quality"""
        if abstained:
            return 0.0
        
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Check for keyword presence in answer
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in answer_lower)
        keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # Length penalty for very short answers
        length_score = min(1.0, len(answer.split()) / 20.0)  # Prefer answers with at least 20 words
        
        # Combine scores
        return (keyword_score * 0.7 + length_score * 0.3)
    
    def run_evaluation(self, questions_file: str = "test_questions.json") -> pd.DataFrame:
        """Run full evaluation on all test questions"""
        questions = self.load_test_questions(questions_file)
        modes = ['baseline', 'hybrid', 'learned']
        
        results = []
        
        print("Running evaluation...")
        for question_data in questions:
            print(f"Evaluating Q{question_data['id']}: {question_data['question'][:50]}...")
            
            for mode in modes:
                try:
                    result = self.evaluate_question(question_data, mode)
                    result.update({
                        'difficulty': question_data['difficulty'],
                        'category': question_data['category']
                    })
                    results.append(result)
                    print(f"  {mode}: relevance={result['relevance_score']:.3f}, quality={result['answer_quality']:.3f}")
                except Exception as e:
                    print(f"  {mode}: ERROR - {e}")
                    results.append({
                        'question_id': question_data['id'],
                        'question': question_data['question'],
                        'mode': mode,
                        'abstained': True,
                        'abstain_reason': f"Error: {e}",
                        'top_score': 0.0,
                        'relevance_score': 0.0,
                        'answer_quality': 0.0,
                        'response_time': 0.0,
                        'num_contexts': 0,
                        'difficulty': question_data['difficulty'],
                        'category': question_data['category']
                    })
        
        return pd.DataFrame(results)
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate evaluation report"""
        report = []
        report.append("# Mini RAG Evaluation Report\n")
        
        # Overall metrics
        report.append("## Overall Performance\n")
        
        # Group by mode
        mode_stats = df.groupby('mode').agg({
            'relevance_score': 'mean',
            'answer_quality': 'mean',
            'top_score': 'mean',
            'abstained': 'sum',
            'response_time': 'mean'
        }).round(3)
        
        report.append("| Mode | Avg Relevance | Avg Quality | Avg Top Score | Abstentions | Avg Response Time |")
        report.append("|------|---------------|-------------|---------------|-------------|------------------|")
        
        for mode in ['baseline', 'hybrid', 'learned']:
            if mode in mode_stats.index:
                stats = mode_stats.loc[mode]
                report.append(f"| {mode.capitalize()} | {stats['relevance_score']:.3f} | {stats['answer_quality']:.3f} | {stats['top_score']:.3f} | {int(stats['abstained'])}/8 | {stats['response_time']:.3f}s |")
        
        report.append("\n")
        
        # Question-by-question results
        report.append("## Question-by-Question Results\n")
        
        for q_id in sorted(df['question_id'].unique()):
            q_data = df[df['question_id'] == q_id]
            question = q_data.iloc[0]['question']
            
            report.append(f"### Q{q_id}: {question}\n")
            
            report.append("| Mode | Relevance | Quality | Top Score | Abstained |")
            report.append("|------|-----------|---------|-----------|-----------|")
            
            for mode in ['baseline', 'hybrid', 'learned']:
                mode_data = q_data[q_data['mode'] == mode]
                if not mode_data.empty:
                    row = mode_data.iloc[0]
                    abstained = "Yes" if row['abstained'] else "No"
                    report.append(f"| {mode.capitalize()} | {row['relevance_score']:.3f} | {row['answer_quality']:.3f} | {row['top_score']:.3f} | {abstained} |")
            
            report.append("\n")
        
        # Improvements analysis
        report.append("## Improvement Analysis\n")
        
        baseline_avg = mode_stats.loc['baseline', 'relevance_score'] if 'baseline' in mode_stats.index else 0
        hybrid_avg = mode_stats.loc['hybrid', 'relevance_score'] if 'hybrid' in mode_stats.index else 0
        learned_avg = mode_stats.loc['learned', 'relevance_score'] if 'learned' in mode_stats.index else 0
        
        if baseline_avg > 0:
            hybrid_improvement = ((hybrid_avg - baseline_avg) / baseline_avg) * 100
            learned_improvement = ((learned_avg - baseline_avg) / baseline_avg) * 100
            
            report.append(f"- **Hybrid improvement over baseline**: {hybrid_improvement:+.1f}%")
            report.append(f"- **Learned improvement over baseline**: {learned_improvement:+.1f}%")
        
        report.append("\n")
        
        return '\n'.join(report)

def main():
    """Run evaluation and generate report"""
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results_df = evaluator.run_evaluation()
    
    # Save results
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\nResults saved to evaluation_results.csv")
    
    # Generate and save report
    report = evaluator.generate_report(results_df)
    with open('evaluation_report.md', 'w') as f:
        f.write(report)
    print("Report saved to evaluation_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    mode_stats = results_df.groupby('mode').agg({
        'relevance_score': 'mean',
        'answer_quality': 'mean',
        'abstained': 'sum'
    }).round(3)
    
    for mode in ['baseline', 'hybrid', 'learned']:
        if mode in mode_stats.index:
            stats = mode_stats.loc[mode]
            print(f"{mode.upper():>8}: Relevance={stats['relevance_score']:.3f}, Quality={stats['answer_quality']:.3f}, Abstained={int(stats['abstained'])}/8")
    
    # Show best performing mode
    best_mode = mode_stats['relevance_score'].idxmax()
    print(f"\nBest performing mode: {best_mode.upper()}")

if __name__ == "__main__":
    main()