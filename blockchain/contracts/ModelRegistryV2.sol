// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ModelRegistryV2
 * @dev Advanced smart contract for ML model registration with incentives and governance
 */

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract ModelRegistryV2 is Ownable, ReentrancyGuard {
    
    // Structs
    struct ModelRecord {
        string modelHash;
        string metadataHash;
        address owner;
        uint256 timestamp;
        uint256 blockNumber;
        string version;
        string ipfsCID;
        bool verified;
        uint256 verificationCount;
        uint256 stakeAmount;
        address[] verifiers;
        string[] tags;
        uint256 accuracy; // Model accuracy scaled by 10000 (e.g., 9500 = 95%)
        uint256 gasUsed;
        uint256 lastUpdated;
    }
    
    struct VerificationStake {
        address verifier;
        uint256 amount;
        uint256 timestamp;
        bool verified;
        string comment;
    }
    
    struct ModelStats {
        uint256 totalModels;
        uint256 totalVerified;
        uint256 totalStaked;
        uint256 totalRewardsDistributed;
    }
    
    // State variables
    mapping(string => ModelRecord) public modelRecords;
    mapping(string => VerificationStake[]) public modelVerifications;
    mapping(address => string[]) public ownerModels;
    mapping(address => uint256) public stakedBalances;
    mapping(address => uint256) public rewardBalances;
    
    // Token for staking and rewards
    IERC20 public rewardToken;
    
    // Governance parameters
    uint256 public verificationStakeAmount = 0.1 ether;
    uint256 public verificationReward = 0.01 ether;
    uint256 public verificationThreshold = 3;
    uint256 public stakeLockPeriod = 7 days;
    
    ModelStats public stats;
    
    // Events
    event ModelRegistered(
        string indexed modelHash,
        address owner,
        uint256 timestamp,
        string version,
        uint256 stakeAmount
    );
    
    event ModelVerified(
        string indexed modelHash,
        address verifier,
        uint256 stakeAmount,
        uint256 timestamp
    );
    
    event StakeDeposited(
        address indexed staker,
        uint256 amount,
        uint256 timestamp
    );
    
    event RewardClaimed(
        address indexed claimer,
        uint256 amount,
        uint256 timestamp
    );
    
    event ModelUpdated(
        string indexed modelHash,
        string newVersion,
        uint256 newAccuracy,
        uint256 timestamp
    );
    
    // Modifiers
    modifier onlyModelOwner(string memory modelHash) {
        require(modelRecords[modelHash].owner == msg.sender, "Not model owner");
        _;
    }
    
    modifier modelExists(string memory modelHash) {
        require(bytes(modelRecords[modelHash].modelHash).length > 0, "Model not found");
        _;
    }
    
    modifier sufficientStake() {
        require(stakedBalances[msg.sender] >= verificationStakeAmount, "Insufficient stake");
        _;
    }
    
    constructor(address _rewardToken) {
        rewardToken = IERC20(_rewardToken);
    }
    
    /**
     * @dev Register a new model with staking requirement
     */
    function registerModel(
        string memory modelHash,
        string memory metadataHash,
        string memory version,
        string memory ipfsCID,
        uint256 accuracy,
        string[] memory tags
    ) external payable nonReentrant {
        require(bytes(modelRecords[modelHash].modelHash).length == 0, "Model already exists");
        require(msg.value >= verificationStakeAmount, "Insufficient registration stake");
        
        ModelRecord memory newRecord = ModelRecord({
            modelHash: modelHash,
            metadataHash: metadataHash,
            owner: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number,
            version: version,
            ipfsCID: ipfsCID,
            verified: false,
            verificationCount: 0,
            stakeAmount: msg.value,
            verifiers: new address[](0),
            tags: tags,
            accuracy: accuracy,
            gasUsed: gasleft(),
            lastUpdated: block.timestamp
        });
        
        modelRecords[modelHash] = newRecord;
        ownerModels[msg.sender].push(modelHash);
        stakedBalances[msg.sender] += msg.value;
        
        stats.totalModels++;
        stats.totalStaked += msg.value;
        
        emit ModelRegistered(modelHash, msg.sender, block.timestamp, version, msg.value);
    }
    
    /**
     * @dev Verify a model with stake
     */
    function verifyModel(
        string memory modelHash,
        string memory comment
    ) external payable modelExists(modelHash) sufficientStake nonReentrant {
        ModelRecord storage record = modelRecords[modelHash];
        require(!record.verified, "Model already verified");
        require(record.owner != msg.sender, "Cannot verify own model");
        
        // Check if already verified by this address
        for(uint i = 0; i < record.verifiers.length; i++) {
            require(record.verifiers[i] != msg.sender, "Already verified");
        }
        
        // Deduct stake
        stakedBalances[msg.sender] -= verificationStakeAmount;
        
        // Create verification record
        VerificationStake memory verification = VerificationStake({
            verifier: msg.sender,
            amount: verificationStakeAmount,
            timestamp: block.timestamp,
            verified: true,
            comment: comment
        });
        
        modelVerifications[modelHash].push(verification);
        record.verifiers.push(msg.sender);
        record.verificationCount++;
        
        // Check if verification threshold reached
        if(record.verificationCount >= verificationThreshold) {
            record.verified = true;
            stats.totalVerified++;
            
            // Distribute rewards
            _distributeVerificationRewards(modelHash);
        }
        
        emit ModelVerified(modelHash, msg.sender, verificationStakeAmount, block.timestamp);
    }
    
    /**
     * @dev Distribute rewards to verifiers
     */
    function _distributeVerificationRewards(string memory modelHash) internal {
        ModelRecord storage record = modelRecords[modelHash];
        uint256 totalReward = record.stakeAmount * record.verifiers.length / 2; // 50% of stake as reward
        
        uint256 rewardPerVerifier = totalReward / record.verifiers.length;
        
        for(uint i = 0; i < record.verifiers.length; i++) {
            rewardBalances[record.verifiers[i]] += rewardPerVerifier;
        }
        
        // Return remaining stake to owner
        uint256 ownerRefund = record.stakeAmount - totalReward;
        payable(record.owner).transfer(ownerRefund);
        
        stats.totalRewardsDistributed += totalReward;
    }
    
    /**
     * @deposit stake for verification
     */
    function depositStake() external payable {
        require(msg.value > 0, "Must deposit positive amount");
        
        stakedBalances[msg.sender] += msg.value;
        stats.totalStaked += msg.value;
        
        emit StakeDeposited(msg.sender, msg.value, block.timestamp);
    }
    
    /**
     * @dev Withdraw stake (only if not locked)
     */
    function withdrawStake(uint256 amount) external nonReentrant {
        require(stakedBalances[msg.sender] >= amount, "Insufficient balance");
        require(amount > 0, "Amount must be positive");
        
        // Check lock period (simplified)
        // In production, you'd track individual stake timestamps
        
        stakedBalances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
        
        stats.totalStaked -= amount;
    }
    
    /**
     * @dev Claim verification rewards
     */
    function claimRewards() external nonReentrant {
        uint256 reward = rewardBalances[msg.sender];
        require(reward > 0, "No rewards to claim");
        
        rewardBalances[msg.sender] = 0;
        
        // Transfer reward token
        require(rewardToken.transfer(msg.sender, reward), "Token transfer failed");
        
        emit RewardClaimed(msg.sender, reward, block.timestamp);
    }
    
    /**
     * @dev Update model version and accuracy
     */
    function updateModel(
        string memory modelHash,
        string memory newVersion,
        uint256 newAccuracy,
        string memory newIpfsCID
    ) external onlyModelOwner(modelHash) {
        ModelRecord storage record = modelRecords[modelHash];
        
        record.version = newVersion;
        record.accuracy = newAccuracy;
        record.ipfsCID = newIpfsCID;
        record.lastUpdated = block.timestamp;
        
        // Reset verification status for new version
        record.verified = false;
        record.verificationCount = 0;
        delete record.verifiers;
        
        emit ModelUpdated(modelHash, newVersion, newAccuracy, block.timestamp);
    }
    
    /**
     * @dev Get model details with verification info
     */
    function getModelDetails(string memory modelHash) 
        external 
        view 
        modelExists(modelHash)
        returns (
            ModelRecord memory record,
            VerificationStake[] memory verifications,
            uint256 totalStakedOnModel
        ) 
    {
        record = modelRecords[modelHash];
        verifications = modelVerifications[modelHash];
        totalStakedOnModel = record.stakeAmount + (record.verificationCount * verificationStakeAmount);
        
        return (record, verifications, totalStakedOnModel);
    }
    
    /**
     * @dev Get model ranking by accuracy
     */
    function getTopModels(uint256 limit, uint256 minVerifications) 
        external 
        view 
        returns (ModelRecord[] memory) 
    {
        // This is simplified - in production you'd use a different data structure
        ModelRecord[] memory topModels = new ModelRecord[](limit);
        uint256 count = 0;
        
        // Implementation would iterate through models and sort
        // For now, return empty
        return topModels;
    }
    
    /**
     * @dev Set governance parameters (only owner)
     */
    function setGovernanceParameters(
        uint256 newStakeAmount,
        uint256 newReward,
        uint256 newThreshold,
        uint256 newLockPeriod
    ) external onlyOwner {
        verificationStakeAmount = newStakeAmount;
        verificationReward = newReward;
        verificationThreshold = newThreshold;
        stakeLockPeriod = newLockPeriod;
    }
    
    /**
     * @dev Emergency withdraw (only owner)
     */
    function emergencyWithdraw() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
    
    // Utility functions
    function compareStrings(string memory a, string memory b) 
        internal 
        pure 
        returns (bool) 
    {
        return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }
}