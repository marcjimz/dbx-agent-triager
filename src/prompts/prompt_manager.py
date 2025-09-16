import mlflow
import yaml
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

class PromptManager:
    """
    A simple manager class for MLFlow Prompt Registry operations.
    Handles prompt registration from YAML files and loading prompts by name/version.
    """
    
    def __init__(self, catalog: str, schema: str):
        """
        Initialize the PromptManager with Unity Catalog catalog and schema.
        
        Args:
            catalog: Unity Catalog catalog name
            schema: Unity Catalog schema name
        """
        self.catalog = catalog
        self.schema = schema
        self.logger = logging.getLogger(__name__)
        
        # Verify MLFlow version
        try:
            import mlflow.genai
        except ImportError:
            raise ImportError("MLFlow GenAI module not found. Please install: pip install --upgrade 'mlflow[databricks]>=3.1.0'")
    
    def _get_full_name(self, prompt_name: str) -> str:
        """
        Construct the full Unity Catalog name for a prompt.
        
        Args:
            prompt_name: Short name of the prompt
            
        Returns:
            Full name in format: catalog.schema.prompt_name
        """
        return f"{self.catalog}.{self.schema}.{prompt_name}"
    
    def _has_changes(self, current_prompt: Any, new_template: str, new_tags: Dict[str, str]) -> tuple[bool, str]:
        """
        Check if there are changes between current prompt and new configuration.
        
        Args:
            current_prompt: Current PromptVersion object
            new_template: New template string
            new_tags: New tags dictionary
            
        Returns:
            Tuple of (has_changes, reason)
        """
        # Normalize templates for comparison (strip whitespace)
        current_template = current_prompt.template.strip() if current_prompt.template else ""
        new_template_normalized = new_template.strip() if new_template else ""
        
        # Check template changes
        if current_template != new_template_normalized:
            self.logger.debug(f"Template changed. Old length: {len(current_template)}, New length: {len(new_template_normalized)}")
            return True, "template changed"
        
        # Check tag changes (if current prompt has tags attribute)
        if hasattr(current_prompt, 'tags'):
            current_tags = current_prompt.tags or {}
            new_tags = new_tags or {}
            if current_tags != new_tags:
                self.logger.debug(f"Tags changed. Old: {current_tags}, New: {new_tags}")
                return True, "tags changed"
        
        return False, "no changes"
    
    def register_from_yaml(self, yaml_file: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Register prompts from a YAML file declaration.
        
        Expected YAML format:
        ```yaml
        prompts:
          - name: customer_support
            template: |
              You are a helpful assistant. 
              Answer: {{question}}
            commit_message: "Initial customer support prompt"
            tags:
              team: support
              model: gpt-4
            alias: production  # Optional: set an alias after registration
            force_update: false  # Optional: force a new version even if unchanged
        ```
        
        Args:
            yaml_file: Path to YAML file containing prompt definitions
            force_update: If True, always create new versions even if unchanged
            
        Returns:
            Dictionary mapping prompt names to their registered versions or status
        """
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'prompts' not in config:
            raise ValueError("YAML file must contain 'prompts' key")
        
        registered = {}
        
        for prompt_config in config['prompts']:
            # Validate required fields
            if 'name' not in prompt_config or 'template' not in prompt_config:
                self.logger.warning(f"Skipping prompt missing name or template: {prompt_config}")
                continue
            
            prompt_name = prompt_config['name']
            full_name = self._get_full_name(prompt_name)
            new_template = prompt_config['template']
            new_tags = prompt_config.get('tags', {})
            
            # Check if we should force update for this specific prompt
            prompt_force_update = prompt_config.get('force_update', force_update)
            
            try:
                # Try to load the latest version to check for changes
                current_version = None
                has_changes = True  # Default to true if prompt doesn't exist
                
                try:
                    current_version = self.load_prompt(prompt_name)
                    has_changes, change_reason = self._has_changes(current_version, new_template, new_tags)
                    
                    if current_version:
                        self.logger.info(f"Found existing prompt {prompt_name} v{current_version.version}")
                        self.logger.debug(f"Change detection result: {change_reason}")
                except Exception as e:
                    # Prompt doesn't exist yet, so we need to create it
                    self.logger.info(f"Prompt {prompt_name} does not exist, will create new")
                    has_changes = True
                    change_reason = "new prompt"
                
                if not has_changes and not prompt_force_update:
                    # No changes detected and not forcing update
                    registered[prompt_name] = {
                        'status': 'unchanged',
                        'version': current_version.version if current_version else None,
                        'full_name': full_name,
                        'message': 'No changes detected, skipped registration'
                    }
                    
                    # Log warning about no changes
                    self.logger.warning(f"⚠️ No changes detected for {full_name} v{current_version.version if current_version else 'N/A'}, skipping registration")
                    print(f"⚠️  Warning: No changes detected for '{prompt_name}' - keeping existing version {current_version.version if current_version else 'N/A'}")
                    
                    # Still update alias if specified and version exists
                    if 'alias' in prompt_config and current_version:
                        mlflow.genai.set_prompt_alias(
                            name=full_name,
                            alias=prompt_config['alias'],
                            version=current_version.version
                        )
                        self.logger.info(f"Updated alias '{prompt_config['alias']}' for existing {full_name} v{current_version.version}")
                else:
                    # Changes detected or force update
                    reason = f" (reason: {change_reason})" if has_changes else " (forced)"
                    commit_message = prompt_config.get('commit_message', 
                                                      f"{'Force updated' if prompt_force_update and not has_changes else 'Updated'} {prompt_name} from YAML{reason}")
                    
                    # Register the new version
                    prompt_version = mlflow.genai.register_prompt(
                        name=full_name,
                        template=new_template,
                        commit_message=commit_message,
                        tags=new_tags
                    )
                    
                    registered[prompt_name] = {
                        'status': 'updated' if current_version else 'created',
                        'version': prompt_version.version,
                        'full_name': full_name,
                        'previous_version': current_version.version if current_version else None
                    }
                    
                    action = "Updated" if current_version else "Created"
                    self.logger.info(f"{action} {full_name} as version {prompt_version.version}")
                    
                    # Set alias if specified
                    if 'alias' in prompt_config:
                        mlflow.genai.set_prompt_alias(
                            name=full_name,
                            alias=prompt_config['alias'],
                            version=prompt_version.version
                        )
                        self.logger.info(f"Set alias '{prompt_config['alias']}' for {full_name} v{prompt_version.version}")
                
            except Exception as e:
                self.logger.error(f"Failed to process prompt {prompt_name}: {e}")
                registered[prompt_name] = {'status': 'error', 'error': str(e)}
        
        return registered
    
    def load_prompt(self, name: str, version: Optional[Union[str, int]] = None) -> Any:
        """
        Load a prompt by name and version (or alias).
        
        Args:
            name: Short name of the prompt (without catalog.schema prefix)
            version: Version number, alias name, or None for latest.
                    Can be:
                    - Integer version number (e.g., 3)
                    - String version number (e.g., "3")
                    - Alias name (e.g., "production", "staging")
                    - None to load the latest version
        
        Returns:
            PromptVersion object containing the prompt template and metadata
        """
        full_name = self._get_full_name(name)
        
        try:
            if version is None:
                # Load latest version using MlflowClient
                self.logger.info(f"Loading latest version of {full_name}")
                from mlflow import MlflowClient
                client = MlflowClient()
                
                # Search for all versions of this specific prompt
                search_response = client.search_prompt_versions(full_name)
                
                if search_response and search_response.prompt_versions:
                    # Get the highest version number
                    latest_version = max(pv.version for pv in search_response.prompt_versions)
                    self.logger.debug(f"Found latest version: {latest_version}")
                    return mlflow.genai.load_prompt(
                        name_or_uri=full_name,
                        version=latest_version
                    )
                else:
                    raise ValueError(f"No versions found for prompt {full_name}")
                
            elif isinstance(version, str) and not version.isdigit():
                # Treat as alias
                self.logger.info(f"Loading {full_name} with alias '{version}'")
                uri = f"prompts:/{full_name}@{version}"
                return mlflow.genai.load_prompt(name_or_uri=uri)
            else:
                # Treat as version number
                version_num = int(version) if isinstance(version, str) else version
                self.logger.info(f"Loading {full_name} version {version_num}")
                return mlflow.genai.load_prompt(
                    name_or_uri=full_name,
                    version=version_num
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load prompt {full_name} (version={version}): {e}")
            raise
    
    def list_prompts(self) -> list:
        """
        List all prompts in the configured catalog and schema.
        
        Returns:
            List of Prompt objects
        """
        try:
            results = mlflow.genai.search_prompts(
                filter_string=f"catalog = '{self.catalog}' AND schema = '{self.schema}'"
            )
            return list(results)
        except Exception as e:
            self.logger.error(f"Failed to list prompts: {e}")
            raise
    
    def set_alias(self, name: str, alias: str, version: int) -> None:
        """
        Set an alias for a specific prompt version.
        
        Args:
            name: Short name of the prompt
            alias: Alias name (e.g., "production", "staging")
            version: Version number to point the alias to
        """
        full_name = self._get_full_name(name)
        try:
            mlflow.genai.set_prompt_alias(
                name=full_name,
                alias=alias,
                version=version
            )
            self.logger.info(f"Set alias '{alias}' for {full_name} v{version}")
        except Exception as e:
            self.logger.error(f"Failed to set alias: {e}")
            raise
    
    def delete_alias(self, name: str, alias: str) -> None:
        """
        Delete an alias for a prompt.
        
        Args:
            name: Short name of the prompt
            alias: Alias name to delete
        """
        full_name = self._get_full_name(name)
        try:
            mlflow.genai.delete_prompt_alias(
                name=full_name,
                alias=alias
            )
            self.logger.info(f"Deleted alias '{alias}' for {full_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete alias: {e}")
            raise


# Example usage
# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Initialize the manager
#     manager = PromptManager(catalog="mycatalog", schema="myschema")
    
    # Example 1: Register from YAML (with change detection)
    # registered = manager.register_from_yaml("prompts.yaml")
    # print(f"Registered prompts: {registered}")
    #
    # Example 1b: Force update even if no changes
    # registered = manager.register_from_yaml("prompts.yaml", force_update=True)
    
    # Example 2: Load latest version
    # prompt = manager.load_prompt("customer_support")
    # print(f"Template: {prompt.template}")
    
    # Example 3: Load specific version
    # prompt_v2 = manager.load_prompt("customer_support", version=2)
    
    # Example 4: Load by alias
    # prod_prompt = manager.load_prompt("customer_support", version="production")
    
    # Example 5: Use the prompt
    # prompt = manager.load_prompt("customer_support")
    # formatted = prompt.format(question="How do I reset my password?")
    # print(f"Formatted prompt: {formatted}")
    
    # Example 6: List all prompts
    # all_prompts = manager.list_prompts()
    # for p in all_prompts:
    #     print(f"Prompt: {p.name}, Tags: {p.tags}")