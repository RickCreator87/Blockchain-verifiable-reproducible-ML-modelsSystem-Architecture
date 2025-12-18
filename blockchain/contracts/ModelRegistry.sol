// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ModelRegistry
 * @dev Smart contract for registering and verifying ML models on blockchain
 */
contract ModelRegistry {
    
    struct ModelRecord {
        string modelHash;          // SHA256 hash of the model
        string metadataHash;       // Hash of model metadata
        address owner;             // Address that registered the model
        uint256 timestamp;         // Block timestamp of registration
        uint256 blockNumber;       // Block number of registration
        string version;            // Model version (e.g., "v1.0.0")
        string ipfsCID;           // IPFS CID for model storage (optional)
        bool verified;            // Whether the model has been verified
    }
    
    // Mapping from model hash to ModelRecord
    mapping(string => ModelRecord) public modelRecords;
    
    // Mapping from owner address to list of model hashes they own
    mapping(address => string[]) public ownerModels;
    
    // Events
    event ModelRegistered(
        string indexed modelHash,
        address owner,
        uint256 timestamp,
        string version
    );
    
    event ModelVerified(
        string indexed modelHash,
        address verifier,
        uint256 timestamp
    );
    
    event ModelUpdated(
        string indexed modelHash,
        string newVersion,
        uint256 timestamp
    );
    
    // Modifier to check if model already exists
    modifier modelNotExists(string memory modelHash) {
        require(bytes(modelRecords[modelHash].modelHash).length == 0, 
                "Model already registered");
        _;
    }
    
    // Modifier to check if model exists
    modifier modelExists(string memory modelHash) {
        require(bytes(modelRecords[modelHash].modelHash).length > 0, 
                "Model not registered");
        _;
    }
    
    // Modifier to check if caller is model owner
    modifier onlyOwner(string memory modelHash) {
        require(modelRecords[modelHash].owner == msg.sender, 
                "Only model owner can perform this action");
        _;
    }
    
    /**
     * @dev Register a new model on blockchain
     * @param modelHash SHA256 hash of the model file
     * @param metadataHash Hash of model metadata
     * @param version Model version string
     * @param ipfsCID IPFS content identifier (optional)
     */
    function registerModel(
        string memory modelHash,
        string memory metadataHash,
        string memory version,
        string memory ipfsCID
    ) public modelNotExists(modelHash) {
        ModelRecord memory newRecord = ModelRecord({
            modelHash: modelHash,
            metadataHash: metadataHash,
            owner: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number,
            version: version,
            ipfsCID: ipfsCID,
            verified: false
        });
        
        modelRecords[modelHash] = newRecord;
        ownerModels[msg.sender].push(modelHash);
        
        emit ModelRegistered(modelHash, msg.sender, block.timestamp, version);
    }
    
    /**
     * @dev Verify a registered model (can be called by anyone)
     * @param modelHash Hash of the model to verify
     */
    function verifyModel(string memory modelHash) 
        public 
        modelExists(modelHash) 
    {
        modelRecords[modelHash].verified = true;
        emit ModelVerified(modelHash, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Update model version (only owner)
     * @param modelHash Current model hash
     * @param newVersion New version string
     * @param newModelHash Hash of new model version
     * @param newMetadataHash Hash of new metadata
     */
    function updateModelVersion(
        string memory modelHash,
        string memory newVersion,
        string memory newModelHash,
        string memory newMetadataHash,
        string memory ipfsCID
    ) public modelExists(modelHash) onlyOwner(modelHash) {
        require(!compareStrings(modelHash, newModelHash),
                "New model hash must be different");
        
        // Create new record for updated model
        ModelRecord memory updatedRecord = ModelRecord({
            modelHash: newModelHash,
            metadataHash: newMetadataHash,
            owner: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number,
            version: newVersion,
            ipfsCID: ipfsCID,
            verified: false
        });
        
        modelRecords[newModelHash] = updatedRecord;
        ownerModels[msg.sender].push(newModelHash);
        
        emit ModelUpdated(modelHash, newVersion, block.timestamp);
    }
    
    /**
     * @dev Get model record by hash
     * @param modelHash Hash of the model
     * @return ModelRecord structure
     */
    function getModelRecord(string memory modelHash) 
        public 
        view 
        modelExists(modelHash)
        returns (
            string memory,
            string memory,
            address,
            uint256,
            uint256,
            string memory,
            string memory,
            bool
        )
    {
        ModelRecord memory record = modelRecords[modelHash];
        return (
            record.modelHash,
            record.metadataHash,
            record.owner,
            record.timestamp,
            record.blockNumber,
            record.version,
            record.ipfsCID,
            record.verified
        );
    }
    
    /**
     * @dev Get models owned by an address
     * @param owner Address of the owner
     * @return Array of model hashes
     */
    function getModelsByOwner(address owner) 
        public 
        view 
        returns (string[] memory) 
    {
        return ownerModels[owner];
    }
    
    /**
     * @dev Check if model is registered
     * @param modelHash Hash to check
     * @return True if model is registered
     */
    function isModelRegistered(string memory modelHash) 
        public 
        view 
        returns (bool) 
    {
        return bytes(modelRecords[modelHash].modelHash).length > 0;
    }
    
    /**
     * @dev Get total number of registered models
     * @return Count of registered models
     */
    function getTotalModels() public view returns (uint256) {
        // Note: This is a simplified implementation
        // In production, you'd want to maintain a counter
        return ownerModels[msg.sender].length;
    }
    
    /**
     * @dev Internal helper to compare strings
     */
    function compareStrings(string memory a, string memory b) 
        internal 
        pure 
        returns (bool) 
    {
        return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }
}