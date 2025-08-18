import pickle
import matplotlib.pyplot as plt

# Loading the last saved checkpoint for comparision
with open('/home/6082/Ein/checkpoints/eval_checkpoint_999.pkl', 'rb') as f:
    simple_results = pickle.load(f)
with open('/home/6082/Ein/checkpoints/cot_eval_checkpoint_999.pkl', 'rb') as f:
    cot_results = pickle.load(f)

# 
simple_correct = simple_results["correct"]
simple_total = simple_results["total"]
simple_acc = (simple_correct / simple_total) * 100

cot_correct = cot_results["correct"]
cot_total = cot_results["total"]
cot_acc = (cot_correct / cot_total) * 100

# Create bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(['Simple Prompt', 'Chain-of-Thought'], [simple_acc, cot_acc], color=['blue', 'orange'], width=0.4)


plt.title('Different prompts')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  

# Add text on bars
for bar, acc, correct, total in zip(bars, [simple_acc, cot_acc], [simple_correct, cot_correct], [simple_total, cot_total]):
    plt.text(bar.get_x() + bar.get_width()/2, acc/2, f'{acc:.1f}%\n({correct}/{total})', ha='center', color='white')

plt.savefig('prompt_comparison.png')
plt.show()
