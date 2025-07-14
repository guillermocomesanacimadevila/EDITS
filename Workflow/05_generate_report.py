import yaml
import json
import os

# Load your HTML template (can be triple-quoted string or read from a file)
with open('Frontend/report_template.html') as f:
    html_template = f.read()

# Example: load config and metrics
with open('path/to/run_config.yaml') as f:
    config = yaml.safe_load(f)
with open('path/to/metrics.json') as f:
    metrics = json.load(f)

# Insert dynamic values into template
html = html_template.replace('{{model_id}}', config['name'])
html = html_template.replace('{{crop_size}}', str(config['size']))
html = html.replace('{{accuracy}}', f"{metrics['accuracy']:.2%}")
# etc...

# Optionally: for batch mode, loop and insert rows
batch_results_html = ""
for result in metrics["batch_results"]:
    batch_results_html += f"<tr><td>{result['filename']}</td><td>{result['tap_score']}</td></tr>\n"
html = html.replace('{{batch_results_rows}}', batch_results_html)

# Write out the final report
with open('Frontend/report.html', 'w') as f:
    f.write(html)
