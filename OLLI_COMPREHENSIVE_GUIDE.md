# Olli Voice AI Assistant - Complete Implementation Guide

## Table of Contents

1. [Technical Architecture Overview](#technical-architecture-overview)
2. [Implementation Requirements](#implementation-requirements)
3. [Core Components Breakdown](#core-components-breakdown)
4. [Function System Architecture](#function-system-architecture)
5. [Audio Processing Pipeline](#audio-processing-pipeline)
6. [Context Management System](#context-management-system)
7. [Step-by-Step Integration Guide](#step-by-step-integration-guide)
8. [Configuration Options](#configuration-options)
9. [Best Practices & Patterns](#best-practices--patterns)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Technical Architecture Overview

Olli is a sophisticated voice-enabled AI assistant built on Google's Gemini 2.5 Flash Preview Native Audio Dialog model, designed for construction management but adaptable to any domain.

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Olli Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Voice UI Layer                                            │
│  ├── VoiceAgentButton (React Component)                    │
│  ├── GlobalVoiceAssistant (App Integration)               │
│  └── OlliAudioService (Core Audio Handler)                │
├─────────────────────────────────────────────────────────────┤
│  Function System                                           │
│  ├── Atomic Functions (Core Business Logic)               │
│  ├── Static Knowledge Functions (Help System)             │
│  └── Business Actions (Server-side Operations)            │
├─────────────────────────────────────────────────────────────┤
│  Context & State Management                                │
│  ├── SessionStatusManager (Singleton)                     │
│  ├── Page Context Builder                                  │
│  └── Knowledge Engine                                      │
├─────────────────────────────────────────────────────────────┤
│  Audio Processing                                          │
│  ├── Audio Utils (Encoding/Decoding)                      │
│  ├── Voice Utils (Response Formatting)                    │
│  └── Voice Metering (Usage Tracking)                      │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── Firebase (Conversations, Transcripts)                │
│  └── Business Logic (Projects, Tasks, etc.)               │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Real-time Voice Interaction**: Native audio streaming with Gemini Live API
- **Function Calling**: Dynamic composition of atomic functions for complex tasks
- **Context Awareness**: Page-aware responses with project and user context
- **Session Persistence**: Maintains state across component mount/unmount cycles
- **Multi-language Support**: Built-in internationalization capabilities
- **Role-based Knowledge**: Adaptive responses based on user permissions and experience
- **Cost Optimization**: Built-in usage tracking and quota management

## Implementation Requirements

### Dependencies

```json
{
  "dependencies": {
    "@google/genai": "^0.21.0",
    "firebase-admin": "^12.0.0",
    "next": "^15.0.0",
    "react": "^18.0.0",
    "@tabler/icons-react": "^3.0.0",
    "nanoid": "^5.0.0"
  }
}
```

### Environment Variables

```bash
# Required
NEXT_PUBLIC_GOOGLE_AI_API_KEY=your-gemini-api-key

# Optional Configuration
NEXT_PUBLIC_OLLI_MODEL=LATEST  # Model selection
NEXT_PUBLIC_GEMINI_API_KEY=fallback-key  # Alternative key name

# Firebase (for conversation tracking)
FIREBASE_PROJECT_ID=your-project
FIREBASE_CLIENT_EMAIL=service-account-email
FIREBASE_PRIVATE_KEY=service-account-key
```

### API Key Setup

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Enable "Gemini 2.5 Flash Preview" model
3. Set appropriate quotas and billing

### Browser Requirements

- HTTPS connection (required for microphone access)
- Modern browser with Web Audio API support
- Microphone permissions

## Core Components Breakdown

### 1. OlliAudioService (Core Engine)

```typescript
// /lib/customer-service-orb/OlliAudioService.tsx
export interface OlliAudioServiceRef {
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  isRecording: boolean;
  setMuted: (muted: boolean) => void;
  setVolume: (volume: number) => void;
}

interface OlliAudioServiceProps {
  apiKey?: string;
  language?: string;
  projectId?: string;
  userContext?: CustomerContext;
  pageContext?: OlliPageContext;
  onStatusChange?: (status: string) => void;
  onTranscriptUpdate?: (transcript: TranscriptEntry) => void;
  onError?: (error: string) => void;
}
```

**Key Features:**
- Headless audio processor (no UI)
- Gemini Live API integration
- Function calling orchestration
- Conversation tracking
- Audio context management

**Critical Lifecycle Rules:**
```typescript
// ⚠️ CRITICAL: Client is created ONLY when user clicks START
// Component mount/unmount does NOT touch client or session
// Sessions persist across component remounts

useEffect(() => {
  // ✅ DO: Restore UI state from SessionManager
  // ✅ DO: Subscribe to status changes
  // ❌ DON'T: Call initClients() here
  // ❌ DON'T: Create Gemini client on mount
}, []);
```

### 2. VoiceAgentButton (UI Component)

```typescript
// /components/voice/VoiceAgentButton.tsx
interface VoiceAgentButtonProps {
  user?: {
    id: string;
    name: string;
    email: string;
    role: string;
    company?: string;
  };
  projectId?: string;
  projectName?: string;
  language?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'minimal' | 'floating';
  position?: 'fixed' | 'relative';
  showStatus?: boolean;
  showMuteButton?: boolean;
  onSessionStart?: () => void;
  onSessionEnd?: () => void;
  onTranscriptUpdate?: (transcript: TranscriptEntry) => void;
}
```

**Usage Examples:**

```typescript
// Basic Usage
<VoiceAgentButton
  user={currentUser}
  projectId={currentProjectId}
  onTranscriptUpdate={(transcript) => {
    console.log('Voice input:', transcript.content);
  }}
/>

// Floating Assistant
<VoiceAgentButton
  user={currentUser}
  variant="floating"
  position="fixed"
  size="lg"
  showMuteButton={true}
/>

// Minimal Sidebar Integration
<VoiceAgentButton
  user={currentUser}
  variant="minimal"
  size="sm"
  showStatus={true}
/>
```

### 3. SessionStatusManager (State Persistence)

```typescript
// /lib/customer-service-orb/session-status-manager.ts
class SessionStatusManager {
  // Session state that persists across component lifecycles
  public currentStatus: string;
  public isSessionActive: boolean;
  public session: any; // Gemini session reference
  public transcript: TranscriptEntry[];
  public conversationId: string | null;
  
  // Audio contexts and streams
  public inputAudioContext: AudioContext | null;
  public outputAudioContext: AudioContext | null;
  public mediaStream: MediaStream | null;
  
  // Singleton pattern
  public static getInstance(): SessionStatusManager;
}
```

**Key Benefits:**
- Maintains session across component unmounts
- Preserves audio contexts and streams
- Stores conversation transcripts
- Notifies all listeners of status changes

## Function System Architecture

Olli uses a sophisticated function calling system with atomic composition and dynamic scheduling.

### Function Categories

#### 1. Atomic Functions (Core Business Logic)

```typescript
// /lib/customer-service-orb/atomic-functions.ts
export const atomicFunctions: ExecutableFunction[] = [
  {
    name: 'find_entity',
    description: 'Universal search across all project entities',
    parameters: {
      type: 'object',
      properties: {
        phrase: { type: 'string', description: 'Search phrase' },
        entityType: { type: 'string', enum: ['projects', 'rfis', 'tasks'] }
      }
    },
    execute: async (args, context) => {
      // Implementation using vector search
      return await findEntityByPhrase(args.phrase, {
        entityType: args.entityType,
        userId: context.userId,
        projectId: context.projectId
      });
    }
  }
];
```

#### 2. Static Knowledge Functions (Help System)

```typescript
export const staticKnowledgeFunctions: ExecutableFunction[] = [
  {
    name: 'help_projects',
    description: 'Complete knowledge about creating, viewing, and managing projects',
    parameters: {
      type: 'object',
      properties: {
        focus: {
          type: 'string',
          enum: ['create', 'view', 'update', 'manage', 'all']
        }
      }
    },
    behavior: Behavior.WHEN_IDLE, // Don't interrupt conversation
    responseScheduling: FunctionResponseScheduling.DYNAMIC,
    execute: async (args, context) => {
      const response = await knowledgeEngine.getContextualKnowledge(
        'projects', 
        args.focus, 
        buildKnowledgeContext(context)
      );
      return {
        success: true,
        knowledge: response.primary,
        examples: response.examples,
        nextSteps: response.nextSteps
      };
    }
  }
];
```

#### 3. Business Actions (Server-side Operations)

```typescript
// /app/actions/olli-business-actions.ts
export async function executeOlliAction(
  functionName: string,
  args: any,
  context?: any
): Promise<any> {
  // Server-side authentication and validation
  const user = await getAuthenticatedUser();
  
  switch (functionName) {
    case 'create_project':
      return await createProject({
        name: args.name,
        address: args.address,
        type: args.type,
        creatorRole: args.creatorRole
      });
      
    case 'create_rfi':
      return await createRFI({
        subject: args.subject,
        question: args.question,
        projectId: args.projectId
      });
      
    // ... more business operations
  }
}
```

### Function Scheduling Behaviors

```typescript
const Behavior = {
  IMMEDIATE: 'immediate',    // Execute immediately
  WHEN_IDLE: 'when_idle'     // Execute during conversation pause
};

const FunctionResponseScheduling = {
  IMMEDIATE: 'immediate',    // Respond immediately
  DYNAMIC: 'dynamic',        // Model decides when to respond
  SILENT: 'silent'           // Don't interrupt conversation
};
```

### Function Composition Examples

Olli supports complex multi-function compositions:

```typescript
// User: "Create a new project called Sunset Mall and add a budget item for concrete"

// Gemini automatically composes:
// 1. create_project({ name: "Sunset Mall", ... })
// 2. create_budget_item({ projectId: newProjectId, item: "concrete" })

// The model handles dependencies and data flow between functions
```

## Audio Processing Pipeline

### Audio Flow Architecture

```
Microphone → Audio Context → Script Processor → PCM Encoding → Gemini Live API
                                                                      ↓
Speaker ← Audio Context ← Audio Buffer ← PCM Decoding ← Audio Response
```

### Key Audio Components

#### 1. Audio Utilities

```typescript
// /lib/customer-service-orb/audio-utils.ts

// Convert Float32Array to Gemini-compatible blob
function createBlob(data: Float32Array): Blob {
  const int16 = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    int16[i] = data[i] * 32768; // Convert float32 to int16
  }
  
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// Decode Gemini audio response
async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer>
```

#### 2. Audio Context Management

```typescript
const initAudio = () => {
  // Input audio (microphone)
  sessionManager.inputAudioContext = new AudioContext({sampleRate: 16000});
  sessionManager.inputNode = sessionManager.inputAudioContext.createGain();
  
  // Output audio (speakers)
  sessionManager.outputAudioContext = new AudioContext({sampleRate: 24000});
  sessionManager.outputNode = sessionManager.outputAudioContext.createGain();
  sessionManager.outputNode.connect(sessionManager.outputAudioContext.destination);
};
```

#### 3. Real-time Audio Processing

```typescript
sessionManager.scriptProcessorNode.onaudioprocess = (audioProcessingEvent) => {
  if (!sessionManager.isRecording) return;
  
  const inputBuffer = audioProcessingEvent.inputBuffer;
  const pcmData = inputBuffer.getChannelData(0);
  
  // Send to Gemini Live API
  sessionManager.session.sendRealtimeInput({
    media: createBlob(pcmData)
  });
};
```

### Voice Response Formatting

```typescript
// /lib/customer-service-orb/voice-utils.ts

export function formatFunctionResponseForVoice(response: any): string {
  // Convert technical responses to natural language
  if (response.data && Array.isArray(response.data)) {
    const count = response.data.length;
    return `Found ${count} item${count !== 1 ? 's' : ''}.`;
  }
  
  // Format amounts for voice
  if (response.amount) {
    return formatAmountForVoice(response.amount);
  }
  
  return response.message || "I've completed that action for you.";
}

export function formatAmountForVoice(amount: number): string {
  if (amount < 1000) return `${amount} dollars`;
  if (amount < 1000000) {
    const thousands = Math.floor(amount / 1000);
    return `${thousands} thousand dollars`;
  }
  const millions = (amount / 1000000).toFixed(1);
  return `${millions} million dollars`;
}
```

## Context Management System

### Page Context Building

```typescript
// /lib/utils/page-context-builder.ts
export function buildPageContext({
  pathname,
  searchParams,
  projectId,
  projectName,
  user,
  customTerminology
}): OlliPageContext {
  return {
    page: {
      url: pathname,
      title: getPageTitle(pathname),
      type: getPageType(pathname),
      section: getPageSection(pathname)
    },
    project: projectId ? {
      id: projectId,
      name: projectName,
      role: user?.role
    } : undefined,
    user: {
      id: user?.id,
      name: user?.name,
      role: user?.role,
      permissions: user?.permissions
    },
    state: detectPageState(pathname, searchParams),
    actions: detectAvailableActions(pathname, user?.role)
  };
}
```

### Enhanced Context with Blueprints

```typescript
export async function enrichContextWithBlueprint(
  context: OlliPageContext,
  projectId?: string,
  userId?: string
): Promise<OlliPageContext> {
  if (context.page?.url?.includes('/blueprints/') && projectId) {
    const blueprintId = extractBlueprintId(context.page.url);
    const blueprint = await getBlueprintDetails(blueprintId, projectId, userId);
    
    return {
      ...context,
      blueprintContext: {
        currentBlueprintName: blueprint.name,
        sheetNumber: blueprint.sheetNumber,
        discipline: blueprint.discipline,
        annotations: blueprint.annotations,
        extractedContent: blueprint.extractedContent
      }
    };
  }
  
  return context;
}
```

### Knowledge Context Integration

```typescript
// /lib/customer-service-orb/knowledge-engine.ts
export interface KnowledgeContext {
  user: {
    uid: string;
    email: string;
    role: string;
    secondaryRoles?: string[];
    experience?: 'novice' | 'intermediate' | 'expert';
  };
  page?: {
    current: string;
    availableActions: string[];
    dataPresent: boolean;
    section?: string;
  };
  project?: {
    id: string;
    name: string;
    type: string;
    phase?: string;
  };
}

export async function getContextualKnowledge(
  topic: string,
  focus: string,
  context: KnowledgeContext
): Promise<KnowledgeResponse> {
  // Role-based knowledge adaptation
  const roleKnowledge = knowledgeData.roleKnowledge[context.user.role];
  
  // Experience-level adaptation
  if (context.user.experience === 'novice') {
    return simplifyKnowledge(roleKnowledge[topic][focus]);
  }
  
  // Context-aware examples
  const examples = getProjectExamples(context.project?.type, topic);
  
  return {
    primary: roleKnowledge[topic][focus],
    contextual: getPageSpecificGuidance(context.page?.current),
    examples,
    nextSteps: getNextSteps(topic, focus, context)
  };
}
```

## Step-by-Step Integration Guide

### Step 1: Basic Setup

```typescript
// 1. Install dependencies
npm install @google/genai firebase-admin @tabler/icons-react nanoid

// 2. Add environment variables
// .env.local
NEXT_PUBLIC_GOOGLE_AI_API_KEY=your-gemini-api-key

// 3. Copy core files to your project
/lib/customer-service-orb/
├── OlliAudioService.tsx
├── session-status-manager.ts
├── audio-utils.ts
├── voice-utils.ts
├── atomic-functions.ts
└── knowledge-engine.ts

/components/voice/
├── VoiceAgentButton.tsx
└── GlobalVoiceAssistant.tsx
```

### Step 2: Basic Integration

```typescript
// app/layout.tsx or dashboard layout
import { GlobalVoiceAssistant } from '@/components/voice/GlobalVoiceAssistant';

export default function DashboardLayout({ children }) {
  const user = useCurrentUser();
  
  return (
    <div className="dashboard-layout">
      {children}
      
      {/* Add floating voice assistant */}
      <GlobalVoiceAssistant
        user={user}
        variant="floating"
        showTranscript={false}
      />
    </div>
  );
}
```

### Step 3: Add Business Functions

```typescript
// Define your domain-specific functions
export const myBusinessFunctions = [
  {
    name: 'get_customers',
    description: 'Retrieve customer list with optional filters',
    parameters: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['active', 'inactive', 'all'] },
        limit: { type: 'number', description: 'Maximum customers to return' }
      }
    },
    execute: async (args, context) => {
      const { getCustomers } = await import('@/app/actions/customers');
      return await getCustomers({
        status: args.status || 'active',
        limit: args.limit || 10,
        userId: context.userId
      });
    }
  }
];

// Add to atomic functions
export const atomicFunctions = [
  ...myBusinessFunctions,
  // ... other functions
];
```

### Step 4: Customize Knowledge System

```typescript
// lib/customer-service-orb/knowledge-data.json
{
  "roleKnowledge": {
    "admin": {
      "customers": {
        "view": "As an admin, you can view all customer accounts...",
        "create": "To create a new customer account...",
        "manage": "Customer management includes..."
      }
    },
    "user": {
      "customers": {
        "view": "You can view customers you have access to...",
        "create": "Contact your admin to create new customers..."
      }
    }
  },
  "quickAnswers": {
    "how to add customer": "Say 'create new customer' and I'll guide you through the process",
    "customer permissions": "Customer access is controlled by your role and assigned accounts"
  }
}
```

### Step 5: Add Server Actions

```typescript
// app/actions/my-business-actions.ts
'use server';

export async function executeMyBusinessAction(
  functionName: string,
  args: any,
  context?: any
): Promise<any> {
  const user = await getAuthenticatedUser();
  
  switch (functionName) {
    case 'create_customer':
      if (!args.name || !args.email) {
        return {
          success: false,
          needsMoreInfo: true,
          message: "I need the customer's name and email address",
          missingParams: ['name', 'email']
        };
      }
      
      return await createCustomer({
        name: args.name,
        email: args.email,
        phone: args.phone,
        company: args.company
      });
      
    default:
      return {
        success: false,
        error: `Unknown function: ${functionName}`
      };
  }
}
```

### Step 6: Add Context Awareness

```typescript
// Customize page context for your app
export function buildMyPageContext(pathname: string, params: any) {
  return {
    page: {
      url: pathname,
      title: getMyPageTitle(pathname),
      type: getMyPageType(pathname)
    },
    // Add your specific context
    customer: params.customerId ? {
      id: params.customerId,
      name: params.customerName
    } : undefined,
    // ... other context
  };
}
```

## Configuration Options

### Model Selection

```typescript
// Environment variable options
NEXT_PUBLIC_OLLI_MODEL=LATEST          // gemini-2.5-flash-native-audio-preview-09-2025
NEXT_PUBLIC_OLLI_MODEL=THINKING        // gemini-2.5-flash-exp-native-audio-thinking-dialog
NEXT_PUBLIC_OLLI_MODEL=HALF_CASCADE    // gemini-2.5-flash-preview-native-audio-dialog
NEXT_PUBLIC_OLLI_MODEL=LIVE_2_5_FLASH  // gemini-live-2.5-flash-preview
NEXT_PUBLIC_OLLI_MODEL=LIVE_2_0_FLASH  // gemini-2.0-flash-live-001
```

### Voice Configuration

```typescript
const config = {
  responseModalities: [Modality.AUDIO],
  speechConfig: {
    voiceConfig: {
      prebuiltVoiceConfig: {
        voiceName: 'Orus'  // Options: Orus, Nova, etc.
      }
    }
  },
  systemInstruction: customSystemPrompt,
  tools: [{ functionDeclarations: allFunctions }],
  toolConfig: {
    functionCallingConfig: {
      mode: "AUTO"  // AUTO, ANY, NONE
    }
  }
};
```

### Usage Tracking

```typescript
// lib/gemini-voice-metering.ts
const VOICE_PRICING = {
  inputPerSecond: (32 * 3.00) / 1_000_000,   // $0.000096/second
  outputPerSecond: (32 * 12.00) / 1_000_000, // $0.000384/second
};

// Quota limits by tier
const QUOTA_LIMITS = {
  free: { daily: 1.00, monthly: 10.00 },      // $1/day, $10/month
  starter: { daily: 5.00, monthly: 50.00 },   // $5/day, $50/month
  pro: { daily: 25.00, monthly: 250.00 },     // $25/day, $250/month
  enterprise: { daily: 100.00, monthly: 1000.00 } // $100/day, $1000/month
};
```

### Language Support

```typescript
// Multi-language greetings
const getGreeting = (lang: string, timeOfDay: number) => {
  if (lang?.startsWith('de')) {
    return timeOfDay < 12 ? 'Guten Morgen' : 'Guten Tag';
  } else if (lang?.startsWith('es')) {
    return timeOfDay < 12 ? 'Buenos días' : 'Buenas tardes';
  } else if (lang?.startsWith('fr')) {
    return timeOfDay < 12 ? 'Bonjour' : 'Bon après-midi';
  }
  return timeOfDay < 12 ? 'Good morning' : 'Good afternoon';
};
```

## Best Practices & Patterns

### 1. Function Design Patterns

#### Atomic Function Pattern
```typescript
// ✅ Good: Small, composable functions
{
  name: 'find_entity',
  description: 'Universal search that works across all entity types',
  parameters: {
    type: 'object',
    properties: {
      phrase: { type: 'string' },
      entityType: { type: 'string', enum: ['customers', 'orders', 'products'] }
    }
  }
}

// ❌ Bad: Large, specific functions
{
  name: 'find_customers_with_recent_orders_and_payment_status',
  // Too specific, not composable
}
```

#### Validation Pattern
```typescript
// ✅ Good: Helpful validation with guidance
case 'create_customer':
  const missingParams = [];
  if (!args.name) missingParams.push('name');
  if (!args.email) missingParams.push('email');
  
  if (missingParams.length > 0) {
    return {
      success: false,
      needsMoreInfo: true,
      message: `I need the customer's ${missingParams.join(' and ')}`,
      missingParams,
      guidance: "Try saying: 'Create customer John Smith with email john@example.com'"
    };
  }
```

#### Voice-Friendly Responses
```typescript
// ✅ Good: Natural language responses
return {
  success: true,
  message: `Created customer ${args.name} successfully. They've been sent a welcome email.`
};

// ❌ Bad: Technical responses
return {
  success: true,
  data: { customerId: 'cust_123', status: 'created' }
};
```

### 2. Context Management Patterns

#### Progressive Context Building
```typescript
// Build context progressively as needed
const context = {
  // Always include
  userId: user.id,
  userRole: user.role,
  
  // Add when relevant
  ...(projectId && { projectId, projectName }),
  ...(customerId && { customerId, customerName }),
  
  // Page-specific context
  pageContext: buildPageContext(pathname, searchParams)
};
```

#### Context-Aware Responses
```typescript
const getContextualMessage = (result: any, context: any) => {
  if (context.pageContext?.url?.includes('/customers')) {
    return `Added to your customer list. You can see ${result.name} in the current view.`;
  }
  return `Created ${result.name} successfully.`;
};
```

### 3. Error Handling Patterns

#### Graceful Degradation
```typescript
try {
  const result = await primaryDataSource();
  return { success: true, data: result };
} catch (error) {
  try {
    const fallback = await fallbackDataSource();
    return { 
      success: true, 
      data: fallback,
      message: "Retrieved from backup system"
    };
  } catch (fallbackError) {
    return {
      success: false,
      error: "Service temporarily unavailable",
      userMessage: "I'm having trouble accessing that data right now. Please try again in a moment."
    };
  }
}
```

#### User-Friendly Error Messages
```typescript
const formatErrorForVoice = (error: any) => {
  if (error.code === 'PERMISSION_DENIED') {
    return "You don't have permission to do that. Would you like me to request access for you?";
  }
  
  if (error.code === 'NOT_FOUND') {
    return "I couldn't find that item. Can you provide more details or try a different search?";
  }
  
  return "Something went wrong. Let me try a different approach.";
};
```

### 4. Performance Optimization

#### Function Response Caching
```typescript
import { cache } from 'react';

const getCachedData = cache(async (userId: string, type: string) => {
  return await expensiveDataQuery(userId, type);
});

// Use in function handlers
case 'get_dashboard_data':
  const data = await getCachedData(context.userId, 'dashboard');
  return formatDashboardData(data);
```

#### Batch Operations
```typescript
case 'create_multiple_items':
  const items = args.itemNames.split(',').map(name => name.trim());
  
  // Process in parallel
  const results = await Promise.all(
    items.map(name => createItem({ name, projectId: context.projectId }))
  );
  
  return {
    success: true,
    message: `Created ${results.length} items successfully`,
    details: results.map(r => r.name).join(', ')
  };
```

### 5. Security Patterns

#### Server-Side Authorization
```typescript
'use server';

export async function executeSecureAction(functionName: string, args: any) {
  // Always verify on server side
  const user = await getAuthenticatedUser();
  if (!user) {
    return { success: false, error: 'Authentication required' };
  }
  
  // Check specific permissions
  const hasPermission = await checkPermission(user.id, functionName, args.resourceId);
  if (!hasPermission) {
    return { success: false, error: 'Insufficient permissions' };
  }
  
  // Proceed with action
  return await executeBusinessLogic(functionName, args, user);
}
```

#### Input Sanitization
```typescript
const sanitizeArgs = (args: any) => {
  return {
    name: sanitizeString(args.name),
    email: sanitizeEmail(args.email),
    amount: sanitizeNumber(args.amount),
    date: sanitizeDate(args.date)
  };
};
```

## Troubleshooting Guide

### Common Issues

#### 1. "Olli doesn't respond"

**Symptoms:**
- Microphone icon shows but no response
- Status stuck on "Connecting..."

**Diagnosis:**
```typescript
// Check these in browser console
console.log('API Key present:', !!process.env.NEXT_PUBLIC_GOOGLE_AI_API_KEY);
console.log('Session manager status:', SessionStatusManager.getInstance().getStatus());
console.log('Audio permissions:', await navigator.permissions.query({name: 'microphone'}));
```

**Solutions:**
1. Verify API key is valid and has quota
2. Ensure HTTPS connection
3. Check microphone permissions
4. Verify Gemini model availability

#### 2. "Function not found"

**Symptoms:**
- Olli says "I don't know how to do that"
- Console shows function execution errors

**Diagnosis:**
```typescript
// Check function registration
console.log('Registered functions:', atomicFunctions.map(f => f.name));
console.log('Function exists:', atomicFunctions.find(f => f.name === 'your_function'));
```

**Solutions:**
1. Ensure function is added to `atomicFunctions` array
2. Check function name spelling matches handler
3. Verify server action is properly exported
4. Update function descriptions for better recognition

#### 3. "Slow responses"

**Symptoms:**
- Long delays between speech and response
- Timeout errors

**Solutions:**
```typescript
// Add caching for read operations
const getCachedProjects = cache(async (userId: string) => {
  return await getProjects(userId);
});

// Optimize database queries
case 'get_projects':
  const projects = await getCachedProjects(user.uid);
  return formatProjectsForVoice(projects);
```

#### 4. "Context not working"

**Symptoms:**
- Olli doesn't understand page references
- No project-specific responses

**Diagnosis:**
```typescript
// Check context building
console.log('Page context:', buildPageContext(pathname, searchParams));
console.log('User context:', userContext);
console.log('Project context:', projectId, projectName);
```

**Solutions:**
1. Ensure `pageContext` is passed to `OlliAudioService`
2. Verify project ID is correctly detected
3. Check context builder implementation

### Debugging Tools

#### 1. Function Call Logging
```typescript
// Enable detailed function logging
const handleFunctionCall = async (functionCall: any) => {
  console.log('🔧 [FUNCTION CALL]', {
    name: functionCall.name,
    args: functionCall.args,
    timestamp: new Date().toISOString()
  });
  
  const result = await executeFunction(functionCall.name, functionCall.args, context);
  
  console.log('✅ [FUNCTION RESULT]', {
    name: functionCall.name,
    success: result.success,
    duration: Date.now() - startTime,
    result: result.success ? 'SUCCESS' : result.error
  });
  
  return result;
};
```

#### 2. Audio Debugging
```typescript
// Monitor audio flow
sessionManager.scriptProcessorNode.onaudioprocess = (event) => {
  const inputBuffer = event.inputBuffer;
  const pcmData = inputBuffer.getChannelData(0);
  
  // Check audio levels
  const volume = Math.sqrt(pcmData.reduce((sum, sample) => sum + sample * sample, 0) / pcmData.length);
  
  if (volume > 0.01) {
    console.log('🎤 Audio detected, volume:', volume.toFixed(4));
  }
};
```

#### 3. Context Debugging
```typescript
// Add context logging to system prompt
const debugContext = `
DEBUG CONTEXT:
User: ${userContext?.name} (${userContext?.role})
Project: ${projectId} - ${projectName}
Page: ${pageContext?.page?.url}
Available Actions: ${pageContext?.actions?.map(a => a.name).join(', ')}
`;

// Include in system instruction for transparency
```

### Performance Monitoring

#### Response Time Tracking
```typescript
const performanceTracker = {
  startFunction: (name: string) => {
    console.time(`function_${name}`);
  },
  
  endFunction: (name: string, success: boolean) => {
    console.timeEnd(`function_${name}`);
    
    // Log to analytics
    recordMetric('function_duration', {
      functionName: name,
      success,
      timestamp: Date.now()
    });
  }
};
```

#### Memory Usage Monitoring
```typescript
// Monitor session memory usage
const getSessionMemoryUsage = () => {
  const session = SessionStatusManager.getInstance();
  return {
    transcriptLength: session.transcript.length,
    audioSources: session.audioSources.size,
    hasActiveSession: session.isActive(),
    memoryEstimate: JSON.stringify(session.transcript).length / 1024 // KB
  };
};
```

### Production Deployment Checklist

- [ ] API key is set in production environment
- [ ] HTTPS is enabled
- [ ] All server actions have proper authentication
- [ ] Error messages are user-friendly
- [ ] Functions are tested with voice input
- [ ] Response times are under 2 seconds
- [ ] Fallback handling for API failures
- [ ] Usage tracking is implemented
- [ ] Documentation is updated
- [ ] Monitoring and alerts are configured

---

This comprehensive guide provides everything needed to implement Olli in any project. The modular architecture allows you to adapt the system to any domain by customizing the function system, knowledge base, and context management while keeping the core audio processing and conversation management intact.