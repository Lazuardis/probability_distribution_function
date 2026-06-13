import pandas as pd
from pathlib import Path

data = {
    'event_number': list(range(1, 11)),
    'interarrival_time': [5.2, 3.8, 4.5, 6.1, 2.9, 5.7, 4.4, 3.3, 5.0, 4.8],
    'service_time': [7.1, 6.3, 5.8, 8.0, 6.9, 7.2, 6.7, 7.5, 6.4, 7.0],
}
df = pd.DataFrame(data)
file_path = Path('dummy_queue_data.xlsx')
df.to_excel(file_path, index=False)
print(f'Created {file_path.resolve()}')
