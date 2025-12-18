from web3 import Web3
import json
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

class ContractDeployer:
    """Deploy smart contracts to blockchain"""
    
    def __init__(self, config_path='../config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Web3 connection
        self.rpc_url = os.getenv('ETHEREUM_RPC_URL', 'http://localhost:8545')
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Load account
        self.private_key = os.getenv('PRIVATE_KEY')
        self.account = self.w3.eth.account.from_key(self.private_key)
        
        print(f"Connected to network: {self.w3.is_connected()}")
        print(f"Network ID: {self.w3.eth.chain_id}")
        print(f"Account: {self.account.address}")
        print(f"Balance: {self.w3.from_wei(self.w3.eth.get_balance(self.account.address), 'ether')} ETH")
    
    def compile_contract(self, contract_path):
        """Compile Solidity contract"""
        import solcx
        solcx.install_solc('0.8.19')
        
        with open(contract_path, 'r') as f:
            contract_source = f.read()
        
        compiled_sol = solcx.compile_source(
            contract_source,
            output_values=['abi', 'bin'],
            solc_version='0.8.19'
        )
        
        contract_id, contract_interface = compiled_sol.popitem()
        return contract_interface
    
    def deploy_contract(self, contract_path):
        """Deploy contract to blockchain"""
        print(f"Compiling contract: {contract_path}")
        contract_interface = self.compile_contract(contract_path)
        
        # Get contract ABI and bytecode
        abi = contract_interface['abi']
        bytecode = contract_interface['bin']
        
        # Create contract instance
        Contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Build transaction
        transaction = Contract.constructor().build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': self.config['blockchain']['gas_limit'],
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign transaction
        signed_txn = self.w3.eth.account.sign_transaction(
            transaction, 
            private_key=self.private_key
        )
        
        # Send transaction
        print("Deploying contract...")
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for receipt
        print(f"Transaction hash: {tx_hash.hex()}")
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(
            tx_hash, 
            timeout=120,
            poll_latency=0.1
        )
        
        contract_address = tx_receipt['contractAddress']
        print(f"Contract deployed at: {contract_address}")
        print(f"Gas used: {tx_receipt['gasUsed']}")
        
        # Save deployment info
        deployment_info = {
            'contract_address': contract_address,
            'transaction_hash': tx_hash.hex(),
            'block_number': tx_receipt['blockNumber'],
            'deployer': self.account.address,
            'network_id': self.w3.eth.chain_id,
            'timestamp': self.w3.eth.get_block(tx_receipt['blockNumber'])['timestamp']
        }
        
        with open('deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Update config with contract address
        self.config['blockchain']['contract_address'] = contract_address
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        return contract_address, abi
    
    def verify_deployment(self, contract_address, expected_bytecode):
        """Verify contract deployment"""
        actual_bytecode = self.w3.eth.get_code(contract_address).hex()
        
        # Compare bytecode (remove metadata hash)
        actual_bytecode_clean = actual_bytecode[:-86]  # Remove metadata
        expected_bytecode_clean = expected_bytecode[:-86]
        
        if actual_bytecode_clean == expected_bytecode_clean:
            print("✅ Contract verification successful!")
            return True
        else:
            print("❌ Contract verification failed!")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy smart contract')
    parser.add_argument('--contract', type=str, 
                       default='contracts/ModelRegistry.sol',
                       help='Path to Solidity contract')
    parser.add_argument('--network', type=str, 
                       default='sepolia',
                       help='Blockchain network to deploy to')
    
    args = parser.parse_args()
    
    deployer = ContractDeployer()
    
    try:
        contract_address, abi = deployer.deploy_contract(args.contract)
        
        print("\n" + "="*50)
        print("DEPLOYMENT SUCCESSFUL!")
        print("="*50)
        print(f"Contract Address: {contract_address}")
        print(f"Network: {args.network}")
        print(f"Explorer: https://{args.network}.etherscan.io/address/{contract_address}")
        print("\nNext steps:")
        print("1. Update your .env file with CONTRACT_ADDRESS")
        print("2. Run tests to verify functionality")
        print("3. Register your first model")
        
    except Exception as e:
        print(f"Deployment failed: {e}")