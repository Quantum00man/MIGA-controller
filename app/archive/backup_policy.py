"""Validated backup policy and scheduling projections (no source writes)."""
from datetime import datetime, timezone

DEFAULTS = {'interval_minutes':10, 'pull_on_startup':False, 'quiet_seconds':60,
            'retry_minutes':10, 'retry_max_minutes':60, 'collections_first':True}


def validate(values):
    if not isinstance(values,dict) or set(values)-set(DEFAULTS):
        raise ValueError('Unknown backup setting')
    result=dict(values)
    for key,value in result.items():
        if isinstance(DEFAULTS[key],bool):
            if not isinstance(value,bool):raise ValueError(key+' must be true/false')
        else:
            maximum=86400 if key=='quiet_seconds' else 10080
            minimum=60 if key=='quiet_seconds' else 1
            if type(value) is not int or not minimum<=value<=maximum:
                raise ValueError(f'{key} must be between {minimum} and {maximum}')
    return result


def effective(config,device):
    policy={**DEFAULTS,**config.get('backup_settings',{}),**device.get('backup_overrides',{})}
    if policy['retry_max_minutes']<policy['retry_minutes']:
        raise ValueError('Maximum retry delay must be at least the initial retry delay')
    return policy


def next_pull(config,device,jobs,boot_time,blocked_until=0):
    policy=effective(config,device)
    if not device.get('enabled',True) or not device.get('automatic_backup',False):return None
    history=[j for j in jobs if j['device_id']==device['device_id']]
    if any(j['status'] in {'queued','running'} for j in history):return None
    interval=policy['interval_minutes']*60
    if not history:
        due=boot_time if policy['pull_on_startup'] else boot_time+interval
    else:
        last=datetime.fromisoformat(history[0]['started_at']).timestamp()
        failures=0
        for job in history:
            if job['status']=='failed' or job.get('failed'):failures+=1
            else:break
        delay=min(policy['retry_max_minutes'],policy['retry_minutes']*2**min(max(failures-1,0),20))*60 if failures else interval
        due=last+delay
        if last<boot_time:due=boot_time if policy['pull_on_startup'] else max(due,boot_time+interval)
    return datetime.fromtimestamp(max(due,blocked_until),timezone.utc).isoformat()
