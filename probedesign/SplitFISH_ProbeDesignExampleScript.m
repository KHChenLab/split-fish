%% Split-FISH Probe Library Generation (https://github.com/khchenlab/split-FISH/)

% This script can be used to generate oligo probes for split-FISH.
% The split-FISH publication is at this link: https://www.nature.com/articles/s41592-020-0858-0

% We modified the probe design scripts from the Zhuang Lab github
% https://github.com/ZhuangLab/MERFISH_analysis/blob/master/example_scripts/library_design_example.m
% to 'split' the probes and attach the bridge sequences
% All the required functions are in the Functions folder.

%% INPUT for this script
% % % 1. Path to trDesigner database (this example uses liver).
% % % This database is made using trDesigner object from
% % % https://github.com/ZhuangLab/MERFISH_analysis/tree/master/probe_construction/TRDesigner.m
% % % 
% % % 2. fasta file of mouse transcripts. This is from gencode.
% https://www.gencodegenes.org/
% % % 
% % % 3. List of transcript IDs (in this example, we use the 317 that we used in the split-FISH paper)
% % % 
% % % 4. Bridge sequences in BridgeSequences.xlsx.

%% OUTPUT for this script
% % % The output of this script is 1) a fasta file with the sequences of
% % % the split probes, 2) codebook file, and 3) fasta file with PCR primers

%% Start of script
clearvars % clear other variables 
close all % close other windows

%% NAMES and PATHS to be changed are in this section

% Display the name of the library
libraryName = 'SplitFISH_Library'; % SplitFISH_Library is an example of a library name
disp(['Your library name is ',libraryName])

selpath = uigetdir('C:\','Select folder where the data is saved');
% Make a new folder to save the results
versionNumber = 1; % this is just a number to keep track of the results when the code has been modified
% Choose where to save the results in
analysisSavePath = [selpath,'\v' num2str(versionNumber) '\'];
mkdir(analysisSavePath); 
disp(['Your results will be saved in ',analysisSavePath])

% The transcriptome object contains the information about the gene IDs, transcript ID, and
% the FPKM data.
liverPath = [selpath,'\LiverMixAdult8Weeks_ENCFF844MJFandENCFF271DWG_polyA\'];

% This is downloaded fasta file of mouse transcripts from gencode.
fastaPath = [selpath,'\gencode.vM4.pc_transcripts.fa\gencode.vM4.combined_transcripts.fa'];

% Path to trDesigner object
trDesignerPath = [selpath,'\tissueTr\targetRegionDesignerObj'];

% Read in the target transcript IDs
[~,transcript_xlsID,~] = xlsread([selpath,'\TranscriptID.xlsx']);

% Use only middle column. These are the list of bridge sequences
bridge_xlsFileName = [selpath,'\BridgeSequences.xlsx'];

%% Load transcriptome
% Transcriptome is a class from the Zhuang Lab github page
% https://github.com/ZhuangLab/MERFISH_analysis/blob/master/probe_construction/Transcriptome.m
% User can create their own Transcriptome object or download an example from our website (Large file size).
transcriptomeLiver = Transcriptome.Load(liverPath);

%% Load fasta file for transcripts fasta. Downloaded from https://www.gencodegenes.org/mouse/release_M4.html
disp(['Loading: ' fastaPath]);
sequences = fastaread(fastaPath); % this is a built-in function from MATLAB
disp(['Found ' num2str(length(sequences)) ' sequences']);
headerData = cellfun(@(x) regexp(x,'\|','split'), {sequences.Header}, 'UniformOutput', false);
transcriptIDsHeader = cellfun(@(x) x{1},headerData, 'UniformOutput', false);
geneIDsHeader = cellfun(@(x) x{2},headerData, 'UniformOutput', false);
geneNames = cellfun(@(x) x{6},headerData, 'UniformOutput', false);
disp(['Found ' num2str(length(unique(geneNames))) ' unique gene ids']);
disp(['Found ' num2str(length(unique(transcriptIDsHeader))) ' unique transcript ids']);

%% Create parallel pool
if isempty(gcp('nocreate')) 
    p = parpool(2); 
    % the number of workers in the parallel pool can be changed to
    % more than 2 on some computers
else
    p = gcp;
end
disp('Created parallel pool')

%% Load Target Region Designer object
% The various functions are from the Zhuang Lab's github page
% trDesigner is from  https://github.com/ZhuangLab/MERFISH_analysis/blob/master/probe_construction/TRDesigner.m
% OTTable is from https://github.com/ZhuangLab/MERFISH_analysis/blob/master/probe_construction/OTTable.m
% OTMap2 is from https://github.com/ZhuangLab/MERFISH_analysis/blob/master/probe_construction/OTMap2.m
% This section may take a few minutes
% User can create their own trDesigner object or download an example from our website (Large file size).

trDesigner = TRDesigner.Load(trDesignerPath);
if ~isempty(p)
    SetParallel(trDesigner,p); % Assign the current parallel pool to the trDesigner
end
disp('The various transcriptomes have been loaded.')

%% Create target regions for a specific set of probe properties
% This section may take a few minutes to run
% Line 924 in TRDesigner has been changed from parfor to for because some
% laptops may have insufficient memory
% We used a regionLength of 52nt because split probes are constructed using pairs of 25-nt sequences with 2-nt spacing in between the pair
% Quartet repeats (AAAA, TTTT, GGGG and CCCC), KpnI restriction sites (GGTACC and CCATGG) and EcoRI restriction sites (GAATTC and CTTAAG) were set as forbidden sequences
% The transcriptome and rRNA specificity table were calculated using a 15-nucleotide seed

genesChosen = trDesigner.DesignTargetRegions(...
    'threePrimeSpace', 2, ...
    'removeForbiddenSeqs', true, ...
    'regionLength', 52, ...
    'specificity', [0.2 1], ...
    'geneID', transcript_xlsID, ...
    'OTTables', {'rRNA', [0, 0]});
disp('The target regions for the genes have been designed')

%% Load bridge sequences
[~,bridge,~] = xlsread(bridge_xlsFileName);

clear readout leftSplit rightSplit
for j = 1:length(bridge)
    readout{j} = bridge{j}; 
    leftSplit{j} = seqrcomplement(readout{j}(1:9)); % left 9 nucleotides. 
    rightSplit{j} = seqrcomplement(readout{j}(10:end)); % right 9 nucleotides. 
end
disp(['The ',num2str(length(bridge)),' bridge sequences have been loaded'])

%% Output a fasta file for analysis of the data to compare with FPKM values after the experiment
% This fasta file has the binary codeword, gene ID, transcript ID, and expected FPKM from database.
NGenes = length(genesChosen);

j = 26; % we use 26 bits
codewords = NchooseKcode(j,2);
finalWords = codewords(randperm(size(codewords,1)), :); % randomly assign codewords to genes

fastaFileName = [libraryName 'E1_codebook.fasta'];
filePath = [analysisSavePath fastaFileName];
warning('off','Bioinfo:fastawrite:AppendToFile');
if exist(filePath, 'file')
    delete(filePath);
    disp(['Deleting ' filePath]);
end

for n = 1:NGenes
    commonName = geneNames(strcmp(transcriptIDsHeader, genesChosen(n).id));
    localGeneName = genesChosen(n).geneName;
    isoform = genesChosen(n).id;
    localAbund = transcriptomeLiver.GetAbundanceByName(localGeneName);
    fastawrite(filePath, num2str(finalWords(n,:)), strcat(char(commonName), ...
        {' '},char(localGeneName),{' '},char(isoform),{' Liver '},num2str(localAbund)));
end

for n = NGenes+1:size(finalWords,1)
    fastawrite(filePath, num2str(finalWords(n,:)), ['Blank' num2str(n-NGenes)]);
end


%% THIS IS MAKING THE PROBE
% EIndex keeps track of the experiments in the probe library. ESize keeps track of the number of oligos in the experiment. Both are used later to assign different primers to different experiments.
% Only one experiment in this example. 

oligos = [];
EIndex = 1;
ESize = 0;
exp = 1;

TileSize = [1:2:72; 2:2:72]; % We use 72 pairs, split into odd and even for each on bits

for a = 1:NGenes
    right = cellfun(@(x) seqrcomplement(x(1:25)), genesChosen(a).sequence, 'UniformOutput', false);
    left = cellfun(@(x) seqrcomplement(x(28:end)), genesChosen(a).sequence, 'UniformOutput', false);
    
    onbits = finalWords(a,:);
    usedbit = find(onbits);
    for b = 1:nnz(onbits)
        % Plus restriction sites
        leftProbe = cellfun(@(x) ['GGTAC ' leftSplit{usedbit(b)} ' TAT ' x ' GAATTC'], left(TileSize(b,:)), 'UniformOutput', false);
        rightProbe = cellfun(@(x) ['GGTACC ' x ' TAT ' rightSplit{usedbit(b)} ' AATTC'], right(TileSize(b,:)), 'UniformOutput', false);
        nameHeader = [libraryName '_E' num2str(exp) '_' char(geneNames(strcmp(transcriptIDsHeader, genesChosen(a).id))) '_' char(genesChosen(a).geneName) '_' char(genesChosen(a).id) '_'];
        probeName = cell(length(TileSize(b,:))*nnz(onbits),1);
        i = 1;
        for k = TileSize(b,:)
            probeName{i} = [nameHeader 'B' num2str(usedbit(b)) '_P1_' num2str(genesChosen(a).startPos(k))];
            probeName{i+length(TileSize(b,:))} = [nameHeader 'B' num2str(usedbit(b)) '_P2_' num2str(genesChosen(a).startPos(k))];
            i = i+1;
        end
        
        allProbes = [leftProbe rightProbe];
        
        for s = EIndex:(EIndex+length(allProbes))-1
            oligos(end+1).Header = probeName{s-EIndex+1};
            oligos(end).Sequence = allProbes{s-EIndex+1};
        end
        EIndex = s;
    end
end

ESize(exp) = length(oligos)-sum(ESize);

%% Design primers
% This section is similar to line 417 of 
% https://github.com/ZhuangLab/MERFISH_analysis/blob/master/example_scripts/library_design_example.m

if isempty(gcp('nocreate'))
    p = parpool(4); % change this to 4 or higher if you can
else
    p = gcp;
end

primersPath = [analysisSavePath 'possible_primers.fasta'];
if ~exist(primersPath)
    
    % Display progress
    PageBreak();
    disp(['Designing primers for ' libraryName]);
    
    % Build Off-Target Table for existing sequences and their reverse
    % complements
    a = cellfun(@(x) x(isletter(x)), {oligos.Sequence}, 'UniformOutput', false);
    seqRcomplement = cellfun(@(x)seqrcomplement(x(~isspace(x))), a, 'UniformOutput', false);
    allSeqs = cellfun(@(x) x(~isspace(x)), a, 'UniformOutput', false);
    allSeqs((end+1):(end+length(seqRcomplement))) = seqRcomplement;
    
    encodingProbeOTTable = OTTable(struct('Sequence', allSeqs), 11, 'verbose', true, ...
        'parallel', p);
    
    % Build primer designer
    prDesigner = PrimerDesigner('numPrimersToGenerate', 1e4, ...
        'seqsToRemove', {'AAAA', 'TTTT', 'GGGG', 'CCCC','GAATTC' ,'CTTAAG'}, ...
        'primerLength', 20, ...
        'OTTables', encodingProbeOTTable, ...
        'OTTableNames', {'encoding'}, ...
        'parallel', p);
    
    % Cut primers
    prDesigner.CutPrimers('Tm', [65 70], ...
        'GC', [.4 .6], ...
        'OTTables', {'encoding', [0,0]});
    prDesigner.RemoveForbiddenSeqs();
    prDesigner.RemoveSelfCompPrimers('homologyMax', 6);
    prDesigner.RemoveHomologousPrimers('homologyMax', 8);
    
    % Write fasta file
    prDesigner.WriteFasta(primersPath);
else
    warning('Found existing primers');
end

%% Checking for dimers and hairpins and add used primers to encoding probes
primers = fastaread(primersPath);
clear primerprops

for i =1:length(primers)
    primerprops(i) = oligoprop(primers(i).Sequence);
end
bad_primers_dimers  = ~cellfun('isempty',{primerprops.Dimers}');
bad_primers_hairpin = ~cellfun('isempty',{primerprops.Hairpins}');
bad_primers = [bad_primers_dimers, bad_primers_hairpin];
good_pos = find(all(~bad_primers,2));
primers = primers(good_pos,:);

finalPrimersPath = [analysisSavePath 'usedPrimers.fasta'];
warning('off','Bioinfo:fastawrite:AppendToFile');
if exist(finalPrimersPath)
    delete(finalPrimersPath);
    disp(['Deleting ' finalPrimersPath]);
end

% Build the final oligos by adding the primers and compiling experiments
finalOligos = [];
for m=1
    for i=sum(ESize(1:m))-ESize(m)+1:sum(ESize(1:m))
        finalOligos(i).Header = oligos(i).Header;
        finalOligos(i).Sequence = [primers(2*m-1).Sequence ' ' ...
            oligos(i).Sequence ' ' ...
            seqrcomplement(primers(2*m).Sequence)];
    end
    primers(2*m-1).Header = ['E' num2str(m) ' Forward Primer'];
    primers(2*m).Header = ['E' num2str(m) ' Reverse Primer'];
    primers(2*m).Sequence = ['TAATACGACTCACTATAGGG' primers(2*m).Sequence]; % T7 promoter sequence for in vitro transcription
    fastawrite (finalPrimersPath,primers(2*m-1))
    fastawrite (finalPrimersPath,primers(2*m))
end

%% Write final fasta
oligosPath = [analysisSavePath 'GIS' libraryName '_oligos.fasta'];
if ~exist(oligosPath)
    
    fastawrite(oligosPath, finalOligos);
    disp(['Wrote: ' oligosPath]);
    
else
    warning('Found existing oligos');
end

%%
fclose('all')
disp('ALL DONE! :)')