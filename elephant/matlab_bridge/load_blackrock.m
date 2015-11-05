%Define python code to run
py_command=sprintf([
    'import sys\n' ...
    'sys.path.insert(0, "/home/denker/Projects/toolboxes/py/python-neo")\n'...
    'import neo\n' ...
    'nikos_session=neo.io.BlackrockIO("/home/denker/DatasetsCached/reachgrasp/DataNikos2/i140701-001")\n'...
    'nikos_blk=nikos_session.read_block(nsx=None,units=[],waveforms=False)\n'...
    'print nikos_blk.name\n'...
    'print nikos_blk.segments[0].spiketrains[0].times\n'...
    'print nikos_blk.segments[0].spiketrains[0].times.units\n']);


disp(sprintf('Preparing python command chain\n-------------------------------------------\n%s', py_command));
disp('');

%Run python code
disp(sprintf('Loading data in Python\n-------------------------------------------'));
py('eval',py_command);

%Get data from python
disp('');
disp(sprintf('Importing Neo block to Matlab\n-------------------------------------------'));
py_import('nikos_blk');

disp(nikos_blk.name);
disp(nikos_blk.segments.name);
disp(nikos_blk.segments.spiketrains.name);
disp(nikos_blk.segments.spiketrains.times(1:10)');
disp(nikos_blk.segments.spiketrains.units);

plot(nikos_blk.segments.spiketrains.times,zeros(length(nikos_blk.segments.spiketrains.times),1),'x');
